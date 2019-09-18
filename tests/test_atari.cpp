#include "gtest/gtest.h"
#include <dqn/atari.h>


TEST(AtariTest, Initialize) {
  dqn::Atari atari("atari_roms/breakout.bin", false);

  ASSERT_EQ(atari.get_action_size(), 4);

  uint8_t* obs_t = new uint8_t[RESIZED_IMAGE_SIZE];
  memset(obs_t, 0, sizeof(uint8_t) * RESIZED_IMAGE_SIZE);
  uint8_t* obs_tp1 = new uint8_t[RESIZED_IMAGE_SIZE];
  memset(obs_tp1, 0, sizeof(uint8_t) * RESIZED_IMAGE_SIZE);
  float rew = -1000.0;
  float ter = 0.0;

  atari.reset(obs_t);
  while (!ter) {
    atari.step(1, obs_tp1, &rew, &ter);
  }

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

  delete[] obs_t;
  delete[] obs_tp1;
}
