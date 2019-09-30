#include <dqn/atari.h>

namespace dqn {

Atari::Atari(const char *rom, bool gui, bool episodic_life, bool random_start,
             default_random_engine rengine) {
  ale_ = make_shared<ALEInterface>(gui);
  ale_->loadROM(rom);
  random_start_ = random_start;
  episodic_life_ = episodic_life;
  rengine_ = rengine;
  legal_actions_ = ale_->getMinimalActionSet();

  was_real_done_ = true;
  lives_ = 0;
  t_ = 0;
  reset_data();
}

Atari::~Atari() {}

void Atari::step(uint8_t act, vector<uint8_t> *obs, float *rew, float *ter) {
  *rew = 0.0;

  for (int i = 0; i < WINDOW_SIZE; ++i) {
    *rew += ale_->act(legal_actions_[act]);
    *ter = game_over() ? 1.0 : 0.0;
    update_current_screen();
    if (*ter)
      break;
  }

  roll_current_obs();
  copy_screens_to_current_obs();
  copy_current_obs_to_input(obs);

  episode_reward_ += *rew;
  ++t_;
  ++t_in_episode_;
}

void Atari::reset(vector<uint8_t> *obs) {
  reset_data();
  reset_game();

  update_current_screen();

  if (random_start_) {
    uniform_int_distribution<> noop_step_dist(0, 30);
    uniform_int_distribution<> random_act_dist(0, get_action_size() - 1);
    int noop_step = noop_step_dist(rengine_);
    for (int i = 0; i < noop_step; ++i) {
      ale_->act(legal_actions_[random_act_dist(rengine_)]);
      update_current_screen();
      if (game_over()) {
        reset_data();
        reset_game();
        update_current_screen();
      }
    }
  }

  roll_current_obs();
  copy_screens_to_current_obs();
  copy_current_obs_to_input(obs);
}

bool Atari::game_over() {
  bool ter = ale_->game_over();
  if (episodic_life_) {
    int current_lives = ale_->lives();
    was_real_done_ = ter;
    if (lives_ > current_lives && current_lives > 0)
      ter = true;
    lives_ = current_lives;
  }
  return ter;
}

void Atari::reset_game() {
  if (episodic_life_) {
    if (was_real_done_)
      ale_->reset_game();
    else
      ale_->act((Action)0);
  } else
    ale_->reset_game();
}

void Atari::copy_current_obs_to_input(vector<uint8_t> *obs) {
  obs->resize(WINDOW_SIZE * RESIZED_IMAGE_SIZE);
  memcpy(obs->data(), current_obs_.data(), WINDOW_SIZE * RESIZED_IMAGE_SIZE);
}

void Atari::update_current_screen() {
  memcpy(last_screen_.data(), current_screen_.data(), RESIZED_IMAGE_SIZE);
  vector<uint8_t> screen;
  ale_->getScreenGrayscale(screen);
  resize(&current_screen_, screen, {RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH},
         {RAW_IMAGE_HEIGHT, RAW_IMAGE_WIDTH});
}

void Atari::copy_screens_to_current_obs() {
  for (int i = 0; i < RESIZED_IMAGE_HEIGHT; ++i) {
    for (int j = 0; j < RESIZED_IMAGE_WIDTH; ++j) {
      int index = i * RESIZED_IMAGE_WIDTH + j;
      uint8_t current_pixel = current_screen_.at(index);
      uint8_t last_pixel = last_screen_.at(index);
      // set maximum pixel between last two screens
      uint8_t max_pixel = max(current_pixel, last_pixel);
      current_obs_[index] = max_pixel;
    }
  }
}

void Atari::reset_data() {
  t_in_episode_ = 0;
  episode_reward_ = 0.0;

  // fill screens with 0
  current_screen_.resize(RESIZED_IMAGE_SIZE);
  memset(current_screen_.data(), 0, current_screen_.size());
  last_screen_.resize(RESIZED_IMAGE_SIZE);
  memset(last_screen_.data(), 0, last_screen_.size());

  // fill obs with 0
  current_obs_.resize(WINDOW_SIZE * RESIZED_IMAGE_SIZE);
  memset(current_obs_.data(), 0, current_obs_.size());
}

void Atari::roll_current_obs() {
  for (int i = 0; i < WINDOW_SIZE - 1; ++i) {
    int src_index = (WINDOW_SIZE - i - 2) * RESIZED_IMAGE_SIZE;
    int dst_index = (WINDOW_SIZE - i - 1) * RESIZED_IMAGE_SIZE;
    memcpy(current_obs_.data() + dst_index, current_obs_.data() + src_index,
           RESIZED_IMAGE_SIZE);
  }
}

}; // namespace dqn
