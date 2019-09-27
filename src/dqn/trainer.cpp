#include <dqn/trainer.h>

namespace dqn {

Trainer::Trainer(shared_ptr<Atari> atari, shared_ptr<Controller> controller,
                 shared_ptr<Evaluator> evaluator, shared_ptr<Monitor> monitor,
                 int final_step, int log_interval, int eval_interval) {
  atari_ = atari;
  controller_ = controller;
  evaluator_ = evaluator;
  monitor_ = monitor;

  final_step_ = final_step;
  log_interval_ = log_interval;
  eval_interval_ = eval_interval;

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

  while (t_ < final_step_) {
    float rew_t = 0.0;
    float ter_t = 0.0;
    atari_->reset(&obs_t);
    while (!ter_t) {
      ++t_;

      // select action
      uint8_t act_tm1 = controller_->act(t_, obs_t);
      memcpy(obs_tm1.data(), obs_t.data(), observation_size);
      atari_->step(act_tm1, &obs_t, &rew_t, &ter_t);

      // store transition
      controller_->store(t_, obs_tm1, act_tm1, rew_t, obs_t, ter_t);

      // update
      float loss = controller_->update(t_);
      if (loss > 0)
        loss_monitor.add(loss);

      if (t_ % log_interval_ == 0) {
        reward_monitor.emit(t_);
        loss_monitor.emit(t_);
      }

      if (t_ % eval_interval_ == 0)
        evaluator_->start(t_);

      if (t_ >= final_step_)
        break;
    }
    reward_monitor.add(atari_->episode_reward());
  }
}

}; // namespace dqn
