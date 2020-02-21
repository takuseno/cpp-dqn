#include <dqn/buffer.h>

namespace dqn {

Buffer::Buffer(int capacity) {
  capacity_ = capacity;
  size_ = 0;
  cursor_ = 0;
  buffer_ = make_shared<deque<TransitionPtr>>();
  random_device seed_gen;
  rengine_ = default_random_engine(seed_gen());
}

Buffer::Buffer(int capacity, default_random_engine rengine) {
  capacity_ = capacity;
  size_ = 0;
  cursor_ = 0;
  buffer_ = make_shared<deque<TransitionPtr>>();
  rengine_ = rengine;
}

void Buffer::add(const vector<uint8_t> &obs_t, uint8_t act_t, float rew_tp1,
                 const vector<uint8_t> &obs_tp1, float ter_tp1) {
  if (size_ < capacity_)
    buffer_->push_back(make_shared<Transition>());

  auto ptr = buffer_->at(cursor_);

  ptr->obs_t.resize(obs_t.size());
  ptr->obs_tp1.resize(obs_tp1.size());
  memcpy(ptr->obs_t.data(), obs_t.data(), obs_t.size());
  memcpy(ptr->obs_tp1.data(), obs_tp1.data(), obs_tp1.size());
  ptr->act_t = act_t;
  ptr->rew_tp1 = rew_tp1;
  ptr->ter_tp1 = ter_tp1;

  ++cursor_;
  if (cursor_ == capacity_)
    cursor_ = 0;

  if (size_ < capacity_)
    ++size_;
}

BatchPtr Buffer::sample(int batch_size) {
  auto batch = make_shared<Batch>();
  batch->obss_t.reserve(batch_size);
  batch->acts_t.reserve(batch_size);
  batch->obss_tp1.reserve(batch_size);
  batch->rews_tp1.reserve(batch_size);
  batch->ters_tp1.reserve(batch_size);

  uniform_int_distribution<> dist(0, size() - 1);
  set<int> indices;
  while (indices.size() < batch_size) {
    int index = dist(rengine_);
    if (indices.find(index) != indices.end())
      continue;
    TransitionPtr ptr = buffer_->at(index);
    batch->obss_t.push_back(&(ptr->obs_t));
    batch->acts_t.push_back(ptr->act_t);
    batch->obss_tp1.push_back(&(ptr->obs_tp1));
    batch->rews_tp1.push_back(ptr->rew_tp1);
    batch->ters_tp1.push_back(ptr->ter_tp1);
    indices.insert(index);
  }
  return batch;
}

}; // namespace dqn
