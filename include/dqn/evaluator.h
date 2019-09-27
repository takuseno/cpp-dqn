#pragma once

#include <dqn/atari.h>
#include <dqn/controllers/controller.h>
#include <dqn/explorations/exploration.h>
#include <dqn/models/dqn.h>
#include <dqn/monitor.h>

using namespace std;

namespace dqn {

class Evaluator {
public:
  Evaluator(shared_ptr<Atari> atari, shared_ptr<Controller> controller,
            shared_ptr<Monitor> monitor, int num_episodes);
  void start(int t);

private:
  int num_episodes_, observation_size_;
  shared_ptr<Atari> atari_;
  shared_ptr<Controller> controller_;
  shared_ptr<MonitorMultiColumnSeries> reward_monitor_;
};

}; // namespace dqn
