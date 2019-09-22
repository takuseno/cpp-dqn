#include "gtest/gtest.h"
#include <dqn/atari.h>


TEST(AtariTest, Initialize) {
  random_device seed;
  default_random_engine engine(seed());
  dqn::Atari atari("atari_roms/breakout.bin", false, true, engine);

  ASSERT_EQ(atari.get_action_size(), 4);

  vector<uint8_t> obs_t;
  vector<uint8_t> obs_tp1;
  float rew = -1000.0;
  float ter = 0.0;

  atari.reset(&obs_t);
  while (!ter) {
    atari.step(1, &obs_tp1, &rew, &ter);
  }

  ASSERT_EQ(obs_t.size(), 4 * 84 * 84);
  ASSERT_EQ(obs_tp1.size(), 4 * 84 * 84);

  bool is_same = true;
  for (int i = 0; i < RESIZED_IMAGE_SIZE; ++i) {
    if (obs_t[i] != obs_tp1[i]) {
      is_same = false;
      break;
    }
  }
  ASSERT_EQ(is_same, false);

  ASSERT_EQ(ter, 1.0);
  ASSERT_NE(rew, -1000.0);
}
