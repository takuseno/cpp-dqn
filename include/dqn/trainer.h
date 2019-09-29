#pragma once

#include <dqn/atari.h>
#include <dqn/controllers/controller.h>
#include <dqn/evaluator.h>
#include <dqn/explorations/exploration.h>
#include <dqn/monitor.h>
#include <memory>

using namespace std;

namespace dqn {

class Trainer {
public:
  Trainer(shared_ptr<Atari> atari, shared_ptr<Controller> controller,
          shared_ptr<Evaluator> evaluator, shared_ptr<Monitor> monitor,
          int final_step, int log_interval, int eval_interval,
          int save_interval);
  void start();

private:
  int t_, final_step_;
  int log_interval_, eval_interval_, save_interval_;
  shared_ptr<Atari> atari_;
  shared_ptr<Controller> controller_;
  shared_ptr<Evaluator> evaluator_;
  shared_ptr<Monitor> monitor_;
};

}; // namespace dqn
