#include <dqn/atari.h>

namespace dqn {

Atari::Atari(const char* rom, bool gui)
{
  ale_ = make_shared<ALEInterface>(gui);
  ale_->loadROM(rom);

  current_screen_ = new uint8_t[RAW_IMAGE_SIZE];

  t_ = 0;
  reset_data();
}


Atari::~Atari()
{
  delete current_screen_;
}


void Atari::step(uint8_t act, uint8_t* obs, float* rew, float* ter)
{
  *rew = ale_->act((Action) act);
  sum_of_rewards_ += *rew;
  *ter = ale_->game_over() ? 1.0 : 0.0;
  get_observation(obs);
}


void Atari::reset(uint8_t* obs)
{
  ale_->reset_game();
  get_observation(obs);
}


void Atari::get_observation(uint8_t* obs)
{
  update_current_screen();
  copy_screen_to_obs(obs);
  reset_data();
}


void Atari::update_current_screen()
{
  vector<uint8_t> screen;
  ale_->getScreenGrayscale(screen);
  memcpy(current_screen_, screen.data(), sizeof(uint8_t) * RAW_IMAGE_SIZE);
}


void Atari::copy_screen_to_obs(uint8_t* obs)
{
  for (int i = 0; i < RESIZED_IMAGE_HEIGHT; ++i) {
    for (int j = 0; j < RESIZED_IMAGE_WIDTH; ++j) {
      int index = i * RESIZED_IMAGE_WIDTH + j;
      int target_y = int((float) i / RESIZED_Y_RATIO);
      int target_x = int((float) j / RESIZED_X_RATIO);
      int target_index = target_y * RAW_IMAGE_WIDTH + target_x;
      obs[index] = current_screen_[target_index];
    }
  }
}


void Atari::reset_data()
{
  t_in_episode_ = 0;
  sum_of_rewards_ = 0.0;
}

};
