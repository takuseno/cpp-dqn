#include "gtest/gtest.h"
#include <dqn/image_utils.h>
#include <random>

using namespace std;

TEST(ImageUtilsTest, Resize) {
  vector<uint8_t> dst;
  vector<uint8_t> src(30 * 30);

  random_device seed;
  default_random_engine engine(seed());
  uniform_int_distribution<> dist(0, 255);
  for (int i = 0; i < 900; ++i) {
    src[i] = dist(engine);
  }

  dqn::resize(&dst, src, {10, 10}, {30, 30});
  for (int i = 0; i < 100; ++i) {
    int x = i % 10;
    int y = i / 10;

    int target_x = 3 * x;
    int target_y = 3 * y;
    int target_index = 30 * target_y + target_x;

    cout << dst[i] << endl;

    ASSERT_EQ(dst[i], src[target_index]);
  }
}
