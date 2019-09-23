#include "gtest/gtest.h"
#include <dqn/exploration.h>

TEST(ExplorationTest, EpsilonGreedy) {
  dqn::EpsilonGreedy exploration(4, 1.0, 0.1, 1000);

  ASSERT_EQ(exploration.epsilon(0), 1.0f);
  ASSERT_EQ(exploration.epsilon(500), 0.55f);
  ASSERT_EQ(exploration.epsilon(1000), 0.1f);
  ASSERT_EQ(exploration.epsilon(2000), 0.1f);

  float q_values[] = {0.1, 0.2, 0.3, 0.2};
  bool is_random = false;
  for (int i = 0; i < 30; ++i) {
    uint8_t action = exploration.sample(q_values, 1000);
    if (action != 2) {
      is_random = true;
      break;
    }
  }
  ASSERT_EQ(is_random, true);
}
