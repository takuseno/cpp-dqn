#include <dqn/atari.h>
#include <dqn/controllers/evaluate.h>
#include <dqn/evaluator.h>
#include <dqn/explorations/epsilon_greedy.h>
#include <dqn/models/dqn.h>
#include <dqn/monitor.h>
#include <gflags/gflags.h>
#include <nbla/global_context.hpp>
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
DEFINE_string(log, "eval", "log directory name");
DEFINE_double(epsilon, 0.05, "final epsilon after schedule");
DEFINE_int32(eval_episodes, 10, "number of episodes to evaluate");

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage("Playing Deep Q-Network powered by NNabla.");
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
      make_shared<Atari>(FLAGS_rom.c_str(), FLAGS_gui, false, true, rengine);

  // epsilon greedy exploration
  auto exploration = make_shared<ConstantEpsilonGreedy>(
      atari->get_action_size(), FLAGS_epsilon, rengine);

  // deep neural network algorithm
  auto model = make_shared<DQN>(atari->get_action_size(), 1, 0.0, 0.0, ctx);

  // load parameters
  if (!FLAGS_load.empty())
    model->load(FLAGS_load.c_str());

  // controllers
  auto controller = make_shared<EvaluateController>(model, exploration);

  // performance monitor
  auto monitor = make_shared<Monitor>(logdir);

  // evaluation loop
  auto evaluator =
      make_shared<Evaluator>(atari, controller, monitor, FLAGS_eval_episodes);
  evaluator->start(0);
}
