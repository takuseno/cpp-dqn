#include "gtest/gtest.h"
#include <dqn/atari.h>


TEST(AtariTest, Initialize) {
  dqn::Atari atari("breakout.bin", false);

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

  ASSERT_EQ(ter, 1.0);
  ASSERT_NE(rew, -1000.0);

  delete[] obs_t;
  delete[] obs_tp1;
}
