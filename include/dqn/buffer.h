#pragma once

#include <deque>
#include <memory>
#include <random>
#include <vector>

using namespace std;

namespace dqn {

struct Batch {
  vector<vector<uint8_t> *> obss_t;
  vector<uint8_t> acts_t;
  vector<float> rews_tp1;
  vector<vector<uint8_t> *> obss_tp1;
  vector<float> ters_tp1;
};

using BatchPtr = shared_ptr<Batch>;

struct Transition {
  vector<uint8_t> obs_t;
  uint8_t act_t;
  float rew_tp1;
  vector<uint8_t> obs_tp1;
  float ter_tp1;
};

using TransitionPtr = shared_ptr<Transition>;

class Buffer {
public:
  Buffer(int capacity);
  Buffer(int capacity, default_random_engine rengine);
  void add(const vector<uint8_t> &obs_t, uint8_t act_t, float rew_tp1,
           const vector<uint8_t> &obs_tp1, float ter_tp1);
  BatchPtr sample(int batch_size);
  int size() { return size_; };

private:
  int capacity_, size_, cursor_;
  default_random_engine rengine_;
  shared_ptr<deque<TransitionPtr>> buffer_;
};

}; // namespace dqn
