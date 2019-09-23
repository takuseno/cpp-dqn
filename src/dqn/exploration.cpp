#include <dqn/exploration.h>

namespace dqn {

EpsilonGreedy::EpsilonGreedy(uint8_t action_size, float start_epsilon,
                             float final_epsilon, int duration) {
  action_size_ = action_size;
  start_epsilon_ = start_epsilon;
  final_epsilon_ = final_epsilon;
  duration_ = duration;
  random_device seed_gen;
  rengine_ = default_random_engine(seed_gen());
}

EpsilonGreedy::EpsilonGreedy(uint8_t action_size, float start_epsilon,
                             float final_epsilon, int duration,
                             default_random_engine rengine) {
  action_size_ = action_size;
  start_epsilon_ = start_epsilon;
  final_epsilon_ = final_epsilon;
  duration_ = duration;
  rengine_ = rengine;
}

uint8_t EpsilonGreedy::sample(float *q_values, int t) {
  std::uniform_real_distribution<> dist(0.0, 1.0);
  // random action
  if (dist(rengine_) < epsilon(t)) {
    std::uniform_int_distribution<> dist(0, action_size_);
    return dist(rengine_);
  }

  // greedy action
  int action = 0;
  float max_q_value = q_values[0];
  for (int i = 1; i < action_size_; ++i) {
    if (q_values[i] > max_q_value) {
      max_q_value = q_values[i];
      action = i;
    }
  }
  return action;
}

float EpsilonGreedy::epsilon(int t) {
  float decay = (float)(duration_ - t) / duration_;
  if (decay < 0.0)
    decay = 0.0;
  float base = start_epsilon_ - final_epsilon_;
  return base * decay + final_epsilon_;
}

}; // namespace dqn
