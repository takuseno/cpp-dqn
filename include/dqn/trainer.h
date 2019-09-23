#pragma once

#include <dqn/atari.h>
#include <dqn/buffer.h>
#include <dqn/model.h>
#include <dqn/exploration.h>
#include <memory>

using namespace std;

namespace dqn {

class Trainer {
public:
  Trainer(shared_ptr<Atari> atari, shared_ptr<Model> model,
          shared_ptr<Buffer> buffer, shared_ptr<EpsilonGreedy> exploration,
          int update_start, int update_interval,
          int target_update_interval, int final_step);
  void start();

private:
  int t_;
  int update_start_, update_interval_, target_update_interval_, final_step_;
  shared_ptr<Atari> atari_;
  shared_ptr<Buffer> buffer_;
  shared_ptr<Model> model_;
  shared_ptr<EpsilonGreedy> exploration_;

  void update();
};

}; // namespace dqn
