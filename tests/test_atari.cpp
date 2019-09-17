#include "gtest/gtest.h"
#include <dqn/atari.h>


TEST(AtariTest, Initialize) {
  dqn::Atari atari("breakout.bin", false);
}
