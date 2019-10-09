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

    int left_x = 3 * x;
    int right_x = min(3 * (x + 1), 30);
    int top_y = 3 * y;
    int bottom_y = min(3 * (y + 1), 30);
    int area_size = (right_x - left_x) * (bottom_y - top_y);

    float pixel = 0;
    for (int y = top_y; y < bottom_y; ++y) {
      for (int x = left_x; x < right_x; ++x) {
        pixel += (float)src[y * 30 + x] / area_size;
      }
    }
    uint8_t diff = abs((uint8_t)pixel - dst[i]);
    ASSERT_EQ(diff < 2, true);
  }
}
