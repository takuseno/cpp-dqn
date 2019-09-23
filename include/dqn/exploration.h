#pragma once

#include <random>

using namespace std;

namespace dqn {

class EpsilonGreedy {
public:
  EpsilonGreedy(uint8_t action_size, float start_epsilon, float final_epsilon,
                int duration);
  EpsilonGreedy(uint8_t action_size, float start_epsilon, float final_epsilon,
                int duration, default_random_engine rengine);
  uint8_t sample(float *q_values, int t);
  float epsilon(int t);

private:
  uint8_t action_size_;
  float start_epsilon_, final_epsilon_;
  int duration_;
  default_random_engine rengine_;
};

}; // namespace dqn
