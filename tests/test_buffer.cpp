#include "gtest/gtest.h"
#include <dqn/buffer.h>


TEST(BufferTest, Size) {
  dqn::Buffer buffer(10);
  ASSERT_EQ(buffer.size(), 0);

  vector<uint8_t> obs_t(4 * 84 * 84);
  vector<uint8_t> obs_tp1(4 * 84 * 84);
  uint8_t act_t = 0;
  float rew_tp1 = 0.0;
  float ter_tp1 = 0.0;

  for (int i = 0; i < 10; ++i) {
    buffer.add(obs_t, act_t, rew_tp1, obs_tp1, ter_tp1);
    ASSERT_EQ(buffer.size(), i + 1);
  }

  buffer.add(obs_t, act_t, rew_tp1, obs_tp1, ter_tp1);
  ASSERT_EQ(buffer.size(), 10);

  dqn::BatchPtr batch = buffer.sample(3);
  ASSERT_EQ(batch->obss_t.size(), 3);
  ASSERT_EQ(batch->obss_tp1.size(), 3);
  ASSERT_EQ(batch->acts_t.size(), 3);
  ASSERT_EQ(batch->rews_tp1.size(), 3);
  ASSERT_EQ(batch->ters_tp1.size(), 3);

  ASSERT_EQ(batch->obss_t.at(0)->size(), 4 * 84 * 84);
  ASSERT_EQ(batch->obss_tp1.at(0)->size(), 4 * 84 * 84);
}
