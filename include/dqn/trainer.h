#pragma once

#include <dqn/atari.h>
#include <dqn/buffer.h>
#include <dqn/evaluator.h>
#include <dqn/explorations/exploration.h>
#include <dqn/models/dqn.h>
#include <dqn/monitor.h>
#include <memory>

using namespace std;

namespace dqn {

class Trainer {
public:
  Trainer(shared_ptr<Atari> atari, shared_ptr<DQN> model,
          shared_ptr<Buffer> buffer, shared_ptr<Exploration> exploration,
          shared_ptr<Evaluator> evaluator, shared_ptr<Monitor> monitor,
          int update_start, int update_interval, int target_update_interval,
          int final_step, int log_interval, int eval_interval);
  void start();

private:
  int t_;
  int update_start_, update_interval_, target_update_interval_, final_step_;
  int log_interval_, eval_interval_;
  shared_ptr<Atari> atari_;
  shared_ptr<Buffer> buffer_;
  shared_ptr<DQN> model_;
  shared_ptr<Exploration> exploration_;
  shared_ptr<Evaluator> evaluator_;
  shared_ptr<Monitor> monitor_;

  float update();
};

}; // namespace dqn
