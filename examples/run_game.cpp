#include <dqn/atari.h>
#include <random>

using namespace std;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    cout << "./run_game <path-to-rom>" << endl;
    return 1;
  }

  dqn::Atari atari(argv[1], true);

  int action_size = atari.get_action_size();

  mt19937 mt(313);
  uniform_int_distribution<> dist(0, action_size - 1);

  int episode = 0;
  vector<uint8_t> obs;
  while (episode < 100) {
    float rew = 0.0;
    float ter = 0.0;
    atari.reset(&obs);
    while (!ter) {
      atari.step(dist(mt), &obs, &rew, &ter);
      uint8_t max_pixel = 0;
      for (int i = 0; i < RESIZED_IMAGE_SIZE; ++i) {
        if (obs[i] > max_pixel)
          max_pixel = obs[i];
      }
      cout << "reward: " << rew << " max pixel: " << (int)max_pixel << endl;
    }
    ++episode;
  }

  return 0;
}
