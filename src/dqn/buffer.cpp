#include <dqn/buffer.h>

namespace dqn {

Buffer::Buffer(int capacity) {
  capacity_ = capacity;
  buffer_ = deque<Experience_t>();
  mt_ = mt19937(313);
}

Buffer::Buffer(int capacity, mt19937 mt) {
  capacity_ = capacity;
  buffer_ = deque<Experience_t>();
  mt_ = mt;
}

Buffer::~Buffer() {
  for (int i = 0; i < buffer_.size(); ++i) {
    deallocate(buffer_[i]);
  }
}

void Buffer::add(const uint8_t *obs_t, uint8_t act_t, float rew_tp1,
                 const uint8_t *obs_tp1, float ter_tp1) {
  int image_size = 4 * 84 * 84;
  uint8_t *dst_obs_t = new uint8_t[image_size];
  uint8_t *dst_obs_tp1 = new uint8_t[image_size];
  memcpy(dst_obs_t, obs_t, sizeof(uint8_t) * image_size);
  memcpy(dst_obs_tp1, obs_tp1, sizeof(uint8_t) * image_size);
  Experience_t exp =
      make_tuple(dst_obs_t, act_t, rew_tp1, dst_obs_tp1, ter_tp1);
  if (buffer_.size() == capacity_)
    buffer_.pop_front();
  buffer_.push_back(exp);
}

Batch_t Buffer::sample(int batch_size) {
  Batch_t batch;
  uniform_int_distribution<> dist(0, size() - 1);
  for (int i = 0; i < batch_size; ++i) {
    int index = dist(mt_);
    get<INDEX_OBS_T>(batch).push_back(get<INDEX_OBS_T>(buffer_[index]));
    get<INDEX_ACT_T>(batch).push_back(get<INDEX_ACT_T>(buffer_[index]));
    get<INDEX_REW_TP1>(batch).push_back(get<INDEX_REW_TP1>(buffer_[index]));
    get<INDEX_OBS_TP1>(batch).push_back(get<INDEX_OBS_TP1>(buffer_[index]));
    get<INDEX_TER_TP1>(batch).push_back(get<INDEX_TER_TP1>(buffer_[index]));
  }
  return batch;
}

int Buffer::size() { return buffer_.size(); }

void Buffer::deallocate(Experience_t exp) {
  delete get<INDEX_OBS_T>(exp);
  delete get<INDEX_OBS_TP1>(exp);
}

}; // namespace dqn
