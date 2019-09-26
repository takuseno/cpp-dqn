#pragma once

#include <dqn/explorations/exploration.h>

namespace dqn {

class ConstantEpsilonGreedy : public Exploration {
public:
  ConstantEpsilonGreedy(uint8_t action_size, float epsilon,
                        default_random_engine rengine);
  virtual uint8_t sample(float *q_values, int t);
  virtual float epsilon(int t) { return epsilon_; }

private:
  uint8_t action_size_;
  float epsilon_;
  default_random_engine rengine_;
};

class LinearDecayEpsilonGreedy : public ConstantEpsilonGreedy {
public:
  LinearDecayEpsilonGreedy(uint8_t action_size, float start_epsilon,
                           float final_epsilon, int duration,
                           default_random_engine rengine);
  virtual float epsilon(int t);

private:
  float start_epsilon_, final_epsilon_;
  int duration_;
};

}; // namespace dqn
