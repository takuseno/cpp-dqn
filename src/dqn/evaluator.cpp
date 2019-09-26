#include <dqn/evaluator.h>

namespace dqn {

Evaluator::Evaluator(shared_ptr<Atari> atari, shared_ptr<DQN> model,
                     shared_ptr<Exploration> exploration,
                     shared_ptr<Monitor> monitor, int num_episodes) {
  atari_ = atari;
  model_ = model;
  exploration_ = exploration;
  reward_monitor_ = make_shared<MonitorMultiColumnSeries>(
      monitor, "eval_reward", num_episodes);
  num_episodes_ = num_episodes;

  observation_size_ = 1;
  auto observation_shape = atari_->get_observation_size();
  for (int i = 0; i < observation_shape.size(); ++i)
    observation_size_ *= observation_shape[i];
}

void Evaluator::start(int t) {
  vector<uint8_t> obs(observation_size_);

  float *q_values = new float[atari_->get_action_size()];

  int episode_count = 0;
  while (episode_count < num_episodes_) {
    float rew = 0.0;
    float ter = 0.0;
    atari_->reset(&obs);
    while (!ter) {
      model_->infer(obs, q_values);
      uint8_t act = exploration_->sample(q_values, t);
      atari_->step(act, &obs, &rew, &ter);
    }
    ++episode_count;
    reward_monitor_->add(atari_->episode_reward());
  }

  reward_monitor_->emit(t);

  delete[] q_values;
}

}; // namespace dqn
