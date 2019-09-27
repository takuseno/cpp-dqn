#include <dqn/controllers/dqn.h>

namespace dqn {

DQNController::DQNController(shared_ptr<Model> model, shared_ptr<Buffer> buffer,
                             shared_ptr<Exploration> exploration,
                             int update_start, int update_interval,
                             int target_update_interval) {
  model_ = model;
  buffer_ = buffer;
  exploration_ = exploration;
  update_start_ = update_start;
  update_interval_ = update_interval;
  target_update_interval_ = target_update_interval;
}

uint8_t DQNController::act(int t, const vector<uint8_t> &obs_t) {
  vector<float> q_values;
  model_->infer(obs_t, &q_values);
  return exploration_->sample(q_values.data(), t);
}

void DQNController::store(int t, const vector<uint8_t> &obs_t, uint8_t act_t,
                          float rew_tp1, const vector<uint8_t> &obs_tp1,
                          float ter_tp1) {
  buffer_->add(obs_t, act_t, rew_tp1, obs_tp1, ter_tp1);
}

float DQNController::update(int t) {
  float loss = -1;
  if (t > update_start_ && t % update_interval_ == 0) {
    BatchPtr batch = buffer_->sample(model_->batch_size());
    loss = model_->train(batch);
  }

  if (t % target_update_interval_ == 0)
    model_->sync_target();

  return loss;
}

}; // namespace dqn
