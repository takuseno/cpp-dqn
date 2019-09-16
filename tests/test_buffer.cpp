#include "gtest/gtest.h"
#include <dqn/buffer.h>


TEST(BufferTest, Size) {
  dqn::Buffer buffer(10);
  ASSERT_EQ(buffer.size(), 0);
}
