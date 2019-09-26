#pragma once

#include <random>

using namespace std;

namespace dqn {

class Exploration {
public:
  virtual uint8_t sample(float *q_values, int t) = 0;
  virtual float epsilon(int t) = 0;
  virtual ~Exploration() {}
};

}; // namespace dqn
