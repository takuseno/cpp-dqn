#include <dqn/atari.h>

namespace dqn {

Atari::Atari(const char *rom, bool gui) {
  ale_ = make_shared<ALEInterface>(gui);
  ale_->loadROM(rom);
  legal_actions_ = ale_->getMinimalActionSet();

  t_ = 0;
  reset_data();
}

Atari::~Atari() {}

void Atari::step(uint8_t act, vector<uint8_t> *obs, float *rew, float *ter) {
  *rew = ale_->act(legal_actions_[act]);
  *ter = ale_->game_over() ? 1.0 : 0.0;
  get_observation(obs);
  sum_of_rewards_ += *rew;
  ++t_;
  ++t_in_episode_;
}

void Atari::reset(vector<uint8_t> *obs) {
  ale_->reset_game();
  get_observation(obs);
  reset_data();
}

void Atari::get_observation(vector<uint8_t> *obs) {
  update_current_screen();
  copy_screen_to_obs(obs);
}

void Atari::update_current_screen() {
  ale_->getScreenGrayscale(current_screen_);
}

void Atari::copy_screen_to_obs(vector<uint8_t> *obs) {
  obs->resize(RESIZED_IMAGE_SIZE);
  for (int i = 0; i < RESIZED_IMAGE_HEIGHT; ++i) {
    for (int j = 0; j < RESIZED_IMAGE_WIDTH; ++j) {
      int index = i * RESIZED_IMAGE_WIDTH + j;
      int target_y = (float)i * RAW_IMAGE_HEIGHT / RESIZED_IMAGE_HEIGHT;
      int target_x = (float)j * RAW_IMAGE_WIDTH / RESIZED_IMAGE_WIDTH;
      int target_index = target_y * RAW_IMAGE_WIDTH + target_x;
      (*obs)[index] = current_screen_.at(target_index);
    }
  }
}

void Atari::reset_data() {
  t_in_episode_ = 0;
  sum_of_rewards_ = 0.0;
}

}; // namespace dqn
