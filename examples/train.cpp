#include <dqn/atari.h>
#include <dqn/buffer.h>
#include <dqn/evaluator.h>
#include <dqn/explorations/epsilon_greedy.h>
#include <dqn/models/dqn.h>
#include <dqn/monitor.h>
#include <dqn/trainer.h>
#include <random>
#include <time.h>

using namespace std;
using namespace dqn;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "./train <path-to-rom>\n");
    return 1;
  }

  // random generator
  random_device seed_gen;
  default_random_engine rengine(seed_gen());

  // logdir
  struct tm *tm;
  char datetime[20];
  time_t timer = time(NULL);
  tm = localtime(&timer);
  strftime(datetime, 20, "%Y%m%d%H%M%S", tm);

  auto ctx = Context({"cpu:float"}, "CpuCachedArray", "0");

  auto atari = make_shared<Atari>(argv[1], false, true, true, rengine);
  auto eval_atari = make_shared<Atari>(argv[1], false, false, true, rengine);

  auto buffer = make_shared<Buffer>(100000);

  auto train_exploration = make_shared<LinearDecayEpsilonGreedy>(
      atari->get_action_size(), 1.0, 0.1, 1000000, rengine);
  auto eval_exploration = make_shared<ConstantEpsilonGreedy>(
      atari->get_action_size(), 0.05, rengine);

  auto model =
      make_shared<DQN>(atari->get_action_size(), 32, 0.99, 0.00025, ctx);

  auto monitor = make_shared<Monitor>(datetime);

  auto evaluator =
      make_shared<Evaluator>(eval_atari, model, eval_exploration, monitor, 10);

  Trainer trainer(atari, model, buffer, train_exploration, evaluator, monitor,
                  50000, 4, 10000, 10000000, 10000, 100000);
  trainer.start();
}
