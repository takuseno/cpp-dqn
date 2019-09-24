#include <dqn/trainer.h>

namespace dqn {

Trainer::Trainer(shared_ptr<Atari> atari, shared_ptr<Model> model,
                 shared_ptr<Buffer> buffer,
                 shared_ptr<EpsilonGreedy> exploration,
                 shared_ptr<Monitor> monitor, int update_start,
                 int update_interval, int target_update_interval,
                 int final_step) {
  atari_ = atari;
  model_ = model;
  buffer_ = buffer;
  exploration_ = exploration;
  monitor_ = monitor;

  update_start_ = update_start;
  update_interval_ = update_interval;
  target_update_interval_ = target_update_interval;
  final_step_ = final_step;

  t_ = 0;
}

void Trainer::start() {
  // Setup Monitors
  MonitorSeries reward_monitor(monitor_, "reward", 100);
  MonitorSeries loss_monitor(monitor_, "loss", 10000);

  int observation_size = 1;
  auto observation_shape = atari_->get_observation_size();
  for (int i = 0; i < observation_shape.size(); ++i)
    observation_size *= observation_shape[i];
  vector<uint8_t> obs_t(observation_size);
  vector<uint8_t> obs_tm1(observation_size);

  float *q_values = new float[atari_->get_action_size()];

  while (t_ < final_step_) {
    float rew_t = 0.0;
    float ter_t = 0.0;
    atari_->reset(&obs_t);
    while (!ter_t) {
      ++t_;

      model_->infer(obs_t, q_values);
      uint8_t act_tm1 = exploration_->sample(q_values, t_ - 1);

      memcpy(obs_tm1.data(), obs_t.data(), observation_size);
      atari_->step(act_tm1, &obs_t, &rew_t, &ter_t);

      buffer_->add(obs_tm1, act_tm1, rew_t, obs_t, ter_t);
      if (t_ > update_start_ && t_ % update_interval_ == 0) {
        float loss = update();
        loss_monitor.add(t_, loss);
      }

      if (t_ % target_update_interval_ == 0)
        model_->sync_target();

      if (t_ >= final_step_)
        break;
    }
    reward_monitor.add(t_, atari_->episode_reward());
  }

  delete[] q_values;
}

float Trainer::update() {
  BatchPtr batch = buffer_->sample(model_->batch_size());
  return model_->train(batch);
}

}; // namespace dqn
