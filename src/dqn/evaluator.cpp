#include <dqn/evaluator.h>

namespace dqn {

Evaluator::Evaluator(shared_ptr<Atari> atari, shared_ptr<Controller> controller,
                     shared_ptr<Monitor> monitor, int num_episodes) {
  atari_ = atari;
  controller_ = controller;
  reward_monitor_ = make_shared<MonitorMultiColumnSeries>(
      monitor, "eval_reward", num_episodes);
  num_episodes_ = num_episodes;
}

void Evaluator::start(int t) {
  vector<uint8_t> obs;

  int episode_count = 0;
  while (episode_count < num_episodes_) {
    float rew = 0.0;
    float ter = 0.0;
    atari_->reset(&obs);
    while (!ter) {
      uint8_t act = controller_->act(t, obs);
      atari_->step(act, &obs, &rew, &ter);
    }
    ++episode_count;
    reward_monitor_->add(atari_->episode_reward());
  }

  reward_monitor_->emit(t);
}

}; // namespace dqn
