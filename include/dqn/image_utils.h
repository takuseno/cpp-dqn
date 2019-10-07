#pragma once

#include <array>
#include <cmath>
#include <dqn/exception.h>
#include <vector>

namespace dqn {

inline void resize(vector<uint8_t> *dst, const vector<uint8_t> &src,
                   const array<int, 2> &dst_shape,
                   const array<int, 2> &src_shape) {
  int src_height = src_shape[0];
  int src_width = src_shape[1];
  int dst_height = dst_shape[0];
  int dst_width = dst_shape[1];

  int dst_size = dst_height * dst_width;
  int src_size = src_height * src_width;
  DQN_CHECK(src.size() == src_size,
            "src_shape does not match with size of src");

  dst->resize(dst_size);

  // bilinear interpolation
  float x_ratio = ((float)src_width) / dst_width;
  float y_ratio = ((float)src_height) / dst_height;
  for (int i = 0; i < dst_size; ++i) {
    // destination position
    int dst_x = i % dst_width;
    int dst_y = i / dst_width;
    int dst_index = dst_y * dst_width + dst_x;

    // source position
    float left_x = floor(dst_x * x_ratio);
    float right_x = floor((dst_x + 1) * x_ratio);
    float top_y = floor(dst_y * y_ratio);
    float bottom_y = floor((dst_y + 1) * y_ratio);

    uint8_t dst_pixel = 0;
    for (int x = left_x; x <= right_x; ++x) {
      float inter_x_ratio = 1.0;
      if (x == left_x)
        inter_x_ratio = x + 1 - ((float)dst_x) * x_ratio;
      else if (x == right_x)
        inter_x_ratio = ((float)(dst_x + 1)) * x_ratio - x;
      for (int y = top_y; y <= bottom_y; ++y) {
        float inter_y_ratio = 1.0;
        if (y == top_y)
          inter_y_ratio = y + 1 - ((float)dst_y) * y_ratio;
        else if (y == bottom_y)
          inter_y_ratio = ((float)(dst_y + 1)) * y_ratio - y;
        uint8_t src_pixel = src.at(int(y * src_width + x));
        dst_pixel += (inter_x_ratio / x_ratio) * (inter_y_ratio / y_ratio) * src_pixel;
      }
    }
    dst->data()[dst_index] = dst_pixel;
  }
}

}; // namespace dqn
