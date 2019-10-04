#include <dqn/atari.h>
#include <dqn/buffer.h>
#include <dqn/controllers/dqn.h>
#include <dqn/controllers/evaluate.h>
#include <dqn/evaluator.h>
#include <dqn/explorations/epsilon_greedy.h>
#include <dqn/models/dqn.h>
#include <dqn/monitor.h>
#include <dqn/trainer.h>
#include <gflags/gflags.h>
#include <nbla/global_context.hpp>
#include <nbla_utils/nnp.hpp>
#include <random>
#include <time.h>

#ifdef GPU
#include <nbla/cuda/cudnn/init.hpp>
#include <nbla/cuda/init.hpp>
#endif

using namespace std;
using namespace dqn;

DEFINE_string(rom, "", "path to ROM file");
DEFINE_string(load, "", "path to parameter file");
DEFINE_bool(gui, false, "render GUI window");
DEFINE_string(log, "experiment", "log directory name");
DEFINE_int32(buffer_size, 100000, "size of replay buffer");
DEFINE_int32(batch_size, 32, "size of batch update");
DEFINE_double(lr, 0.00025, "learning rate");
DEFINE_double(start_epsilon, 1.0, "epsilon at the beggining of training");
DEFINE_double(end_epsilon, 0.1, "final epsilon after schedule");
DEFINE_int32(exploration_duration, 1000000, "step length to decay epsilon");
DEFINE_double(eval_epsilon, 0.05, "epsilon during evaluation");
DEFINE_double(gamma, 0.99, "discount factor");
DEFINE_int32(update_interval, 4, "interval to train");
DEFINE_int32(target_update_interval, 10000, "interval to sync target function");
DEFINE_int32(learning_start, 50000, "step to start learning");
DEFINE_int32(final_step, 10000000, "step to finish training");
DEFINE_int32(log_interval, 10000, "interval to emit log data");
DEFINE_int32(eval_interval, 100000, "interval to perform evaluation");
DEFINE_int32(eval_episodes, 10, "number of episodes to evaluate");
DEFINE_int32(save_interval, 1000000, "interval to save parameters");

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("Deep Q-Network powered by NNabla.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // random generator
  random_device seed_gen;
  default_random_engine rengine(seed_gen());

  // logdir
  struct tm *tm;
  char datetime[20];
  time_t timer = time(NULL);
  tm = localtime(&timer);
  strftime(datetime, 20, "%Y%m%d%H%M%S", tm);
  string logdir = datetime;
  if (!FLAGS_log.empty())
    logdir = FLAGS_log + "_" + logdir;

#ifdef GPU
  nbla::init_cudnn();
  nbla::Context ctx{
      {"cudnn:float", "cuda:float", "cpu:float"}, "CudaCachedArray", "0"};
  cout << "GPU is enabled." << endl;
#else
  nbla::Context ctx{{"cpu:float"}, "CpuCachedArray", "0"};
#endif
  SingletonManager::get<GlobalContext>()->set_current_context(ctx);

  // environments
  auto atari =
      make_shared<Atari>(FLAGS_rom.c_str(), FLAGS_gui, true, true, rengine);
  auto eval_atari =
      make_shared<Atari>(FLAGS_rom.c_str(), false, false, true, rengine);

  // replay buffer
  auto buffer = make_shared<Buffer>(FLAGS_buffer_size);

  // epsilon greedy exploration
  auto exploration = make_shared<LinearDecayEpsilonGreedy>(
      atari->get_action_size(), FLAGS_start_epsilon, FLAGS_end_epsilon,
      FLAGS_exploration_duration, rengine);
  auto eval_exploration = make_shared<ConstantEpsilonGreedy>(
      atari->get_action_size(), FLAGS_eval_epsilon, rengine);

  // deep neural network algorithm
  auto model = make_shared<DQN>(atari->get_action_size(), FLAGS_batch_size,
                                FLAGS_gamma, FLAGS_lr, ctx);

  // controllers
  auto controller = make_shared<DQNController>(
      model, buffer, exploration, FLAGS_learning_start, FLAGS_update_interval,
      FLAGS_target_update_interval);
  auto eval_controller =
      make_shared<EvaluateController>(model, eval_exploration);

  // performance monitor
  auto monitor = make_shared<Monitor>(logdir);

  // evaluation loo
  auto evaluator = make_shared<Evaluator>(eval_atari, eval_controller, monitor,
                                          FLAGS_eval_episodes);

  // training loop
  Trainer trainer(atari, controller, evaluator, monitor, FLAGS_final_step,
                  FLAGS_log_interval, FLAGS_eval_interval, FLAGS_save_interval);
  trainer.start();
}
