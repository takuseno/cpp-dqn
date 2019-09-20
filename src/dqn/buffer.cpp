#include <dqn/buffer.h>

namespace dqn {

Buffer::Buffer(int capacity) {
  capacity_ = capacity;
  size_ = 0;
  buffer_ = make_shared<deque<TransitionPtr>>(capacity);
  mt_ = mt19937(313);
}

Buffer::Buffer(int capacity, mt19937 mt) {
  capacity_ = capacity;
  size_ = 0;
  buffer_ = make_shared<deque<TransitionPtr>>(capacity);
  mt_ = mt;
}

void Buffer::add(const vector<uint8_t> &obs_t, uint8_t act_t, float rew_tp1,
                 const vector<uint8_t> &obs_tp1, float ter_tp1) {
  auto transition = make_shared<Transition>();
  transition->obs_t.reserve(obs_t.size());
  transition->obs_tp1.reserve(obs_tp1.size());
  memcpy(transition->obs_t.data(), obs_t.data(), obs_t.size());
  memcpy(transition->obs_tp1.data(), obs_tp1.data(), obs_tp1.size());
  transition->act_t = act_t;
  transition->rew_tp1 = rew_tp1;
  transition->ter_tp1 = ter_tp1;
  if (size_ == capacity_)
    buffer_->pop_front();
  else
    ++size_;
  buffer_->push_back(transition);
}

BatchPtr Buffer::sample(int batch_size) {
  auto batch = make_shared<Batch>();
  uniform_int_distribution<> dist(0, size() - 1);
  for (int i = 0; i < batch_size; ++i) {
    int index = dist(mt_);
    auto transition = buffer_->at(index);
    batch->obss_t.push_back(&transition->obs_t);
    batch->obss_tp1.push_back(&transition->obs_tp1);
    batch->acts_t.push_back(transition->act_t);
    batch->ters_tp1.push_back(transition->ter_tp1);
    batch->rews_tp1.push_back(transition->rew_tp1);
  }
  return batch;
}

}; // namespace dqn
