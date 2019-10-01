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
  for (int i = 0; i < dst_size; ++i) {
    // destination position
    int dst_x = i % dst_width;
    int dst_y = i / dst_width;
    int dst_index = dst_y * dst_width + dst_x;

    // source position
    float src_x = ((float)dst_x) * src_width / dst_width;
    float src_y = ((float)dst_y) * src_height / dst_height;
    float left_x = floor(src_x);
    float right_x = ceil(src_x);
    float top_y = floor(src_y);
    float bottom_y = ceil(src_y);

    // source pixel values
    float left_top = src.at(int(top_y * src_width + left_x));
    float right_top = src.at(int(top_y * src_width + right_x));
    float left_bottom = src.at(int(bottom_y * src_width + left_x));
    float right_bottom = src.at(int(bottom_y * src_width + right_x));

    // vertical interpolation
    float dy = bottom_y - src_y;
    float left = dy * left_top + (1 - dy) * left_bottom;
    float right = dy * right_top + (1 - dy) * right_bottom;

    // horizontal interpolation and set
    float dx = right_x - src_x;
    dst->data()[dst_index] = dx * left + (1 - dx) * right;
  }
}

}; // namespace dqn
