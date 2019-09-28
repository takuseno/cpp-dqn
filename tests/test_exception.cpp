#include <dqn/exception.h>
#include <gtest/gtest.h>

using namespace dqn;

TEST(ExceptionTest, CheckDQN_CHECK) {
  try {
    DQN_CHECK(false, "test");
  } catch (Exception e) {
    ASSERT_STREQ(e.what(),
                 "false check failed at 7 of test_exception.cpp\ntest\n");
  }
}
