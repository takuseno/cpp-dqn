#pragma once

#include <ale_interface.hpp>
#include <algorithm>
#include <array>
#include <dqn/image_utils.h>
#include <memory>
#include <random>

using namespace std;

#define RAW_IMAGE_WIDTH 160
#define RAW_IMAGE_HEIGHT 210
#define RAW_IMAGE_SIZE 210 * 160
#define RESIZED_IMAGE_WIDTH 84
#define RESIZED_IMAGE_HEIGHT 84
#define RESIZED_IMAGE_SIZE 84 * 84
#define WINDOW_SIZE 4

namespace dqn {

class Atari {
public:
  Atari(const char *rom, bool gui, bool episodic_life, bool random_start,
        default_random_engine rengine);
  ~Atari();
  void step(uint8_t act, vector<uint8_t> *obs, float *rew, float *ter);
  void reset(vector<uint8_t> *obs);
  int get_action_size() { return legal_actions_.size(); }
  array<int, 3> get_observation_size() {
    return {WINDOW_SIZE, RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH};
  }
  float episode_reward() { return episode_reward_; };

private:
  shared_ptr<ALEInterface> ale_;
  int t_;
  int t_in_episode_;
  float episode_reward_;
  bool random_start_;
  bool episodic_life_;
  bool was_real_done_;
  int lives_;
  default_random_engine rengine_;
  vector<uint8_t> current_screen_;
  vector<uint8_t> last_screen_;
  vector<uint8_t> current_obs_;
  ActionVect legal_actions_;

  void reset_data();
  void reset_game();
  bool game_over();
  void get_observation(vector<uint8_t> *obs);
  void update_current_screen();
  void copy_screens_to_current_obs();
  void copy_current_obs_to_input(vector<uint8_t> *obs);
  void roll_current_obs();
};

}; // namespace dqn
