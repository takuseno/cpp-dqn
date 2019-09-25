#pragma once

#include <dqn/atari.h>
#include <dqn/buffer.h>
#include <dqn/exploration.h>
#include <dqn/models/dqn.h>
#include <dqn/monitor.h>
#include <memory>

using namespace std;

namespace dqn {

class Trainer {
public:
  Trainer(shared_ptr<Atari> atari, shared_ptr<DQN> model,
          shared_ptr<Buffer> buffer, shared_ptr<EpsilonGreedy> exploration,
          shared_ptr<Monitor> monitor, int update_start, int update_interval,
          int target_update_interval, int final_step);
  void start();

private:
  int t_;
  int update_start_, update_interval_, target_update_interval_, final_step_;
  shared_ptr<Atari> atari_;
  shared_ptr<Buffer> buffer_;
  shared_ptr<DQN> model_;
  shared_ptr<EpsilonGreedy> exploration_;
  shared_ptr<Monitor> monitor_;

  float update();
};

}; // namespace dqn
