#include <dqn/atari.h>
#include <dqn/buffer.h>
#include <dqn/exploration.h>
#include <dqn/model.h>
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
  auto buffer = make_shared<Buffer>(100000);
  auto exploration =
      make_shared<EpsilonGreedy>(atari->get_action_size(), 1.0, 0.1, 1000000);
  auto model =
      make_shared<Model>(atari->get_action_size(), 32, 0.99, 0.00025, ctx);
  auto monitor = make_shared<Monitor>(datetime);

  Trainer trainer(atari, model, buffer, exploration, monitor, 50000, 4, 10000,
                  10000000);
  trainer.start();
}
