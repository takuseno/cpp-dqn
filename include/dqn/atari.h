#include <ale_interface.hpp>
#include <array>
#include <memory>

using namespace std;

#define RAW_IMAGE_WIDTH 160
#define RAW_IMAGE_HEIGHT 210
#define RAW_IMAGE_SIZE 210 * 160
#define RESIZED_IMAGE_WIDTH 84
#define RESIZED_IMAGE_HEIGHT 84
#define RESIZED_IMAGE_SIZE 84 * 84

namespace dqn {

class Atari {
public:
  Atari(const char *rom, bool gui);
  ~Atari();
  void step(uint8_t act, uint8_t *obs, float *rew, float *ter);
  void reset(uint8_t *obs);
  int get_action_size() { return legal_actions_.size(); }

private:
  shared_ptr<ALEInterface> ale_;
  int t_;
  int t_in_episode_;
  float sum_of_rewards_;
  vector<uint8_t> current_screen_;
  ActionVect legal_actions_;

  void reset_data();
  void get_observation(uint8_t *obs);
  void update_current_screen();
  void copy_screen_to_obs(uint8_t *obs);
};

}; // namespace dqn
