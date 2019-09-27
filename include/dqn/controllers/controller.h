#pragma once

#include <vector>

using namespace std;

namespace dqn {

class Controller {
public:
  virtual ~Controller() {}
  virtual uint8_t act(int t, const vector<uint8_t> &obs_t) = 0;
  virtual void store(int t, const vector<uint8_t> &obs_t, uint8_t act_t,
                     float rew_tp1, const vector<uint8_t> &obs_tp1,
                     float ter_t) = 0;
  virtual float update(int t) = 0;
};

}; // namespace dqn
