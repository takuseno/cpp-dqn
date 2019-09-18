#pragma once

#include <deque>
#include <random>
#include <string.h>
#include <tuple>
#include <vector>

#define INDEX_OBS_T 0
#define INDEX_ACT_T 1
#define INDEX_REW_TP1 2
#define INDEX_OBS_TP1 3
#define INDEX_TER_TP1 4

using namespace std;

namespace dqn {

typedef tuple<vector<const uint8_t *>, vector<uint8_t>, vector<float>,
              vector<const uint8_t *>, vector<float>>
    Batch_t;

typedef tuple<const uint8_t *, uint8_t, float, const uint8_t *, float>
    Experience_t;

class Buffer {
public:
  Buffer(int capacity);
  Buffer(int capacity, mt19937 mt);
  ~Buffer();
  void add(const uint8_t *obs_t, uint8_t act_t, float rew_tp1,
           const uint8_t *obs_tp1, float ter_tp1);
  Batch_t sample(int batch_size);
  int size();

private:
  int capacity_;
  mt19937 mt_;
  deque<Experience_t> buffer_;
  void deallocate(Experience_t exp);
};

}; // namespace dqn
