#pragma once

#include <deque>
#include <random>
#include <memory>
#include <array>
#include <string.h>
#include <vector>
#include <dqn/constants.h>

using namespace std;

namespace dqn {

struct Batch {
  vector<array<uint8_t, OBS_SIZE>*> obss_t;
  vector<uint8_t> acts_t;
  vector<float> rews_tp1;
  vector<array<uint8_t, OBS_SIZE>*> obss_tp1;
  vector<float> ters_tp1;
};

using BatchPtr = shared_ptr<Batch>;

struct Transition {
  array<uint8_t, OBS_SIZE> obs_t;
  uint8_t act_t;
  float rew_tp1;
  array<uint8_t, OBS_SIZE> obs_tp1;
  float ter_tp1;
};

using TransitionPtr = shared_ptr<Transition>;

class Buffer {
public:
  Buffer(int capacity);
  Buffer(int capacity, mt19937 mt);
  void add(const array<uint8_t, OBS_SIZE> &obs_t, uint8_t act_t, float rew_tp1,
           const array<uint8_t, OBS_SIZE> &obs_tp1, float ter_tp1);
  BatchPtr sample(int batch_size);
  int size() { return size_; };

private:
  int capacity_, size_;
  mt19937 mt_;
  shared_ptr<deque<TransitionPtr>> buffer_;
};

}; // namespace dqn
