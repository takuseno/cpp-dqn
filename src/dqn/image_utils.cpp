#include <dqn/image_utils.h>

namespace dqn {

void resize(vector<uint8_t> *dst, const vector<uint8_t> &src,
            const array<int, 2> &dst_shape, const array<int, 2> &src_shape) {
  int src_height = src_shape[0];
  int src_width = src_shape[1];
  int dst_height = dst_shape[0];
  int dst_width = dst_shape[1];

  int dst_size = dst_height * dst_width;
  int src_size = src_height * src_width;
  DQN_CHECK(src.size() == src_size,
            "src_shape does not match with size of src");

  dst->resize(dst_size);

  // area averaging interpolation
  float x_ratio = ((float)src_width) / dst_width;
  float y_ratio = ((float)src_height) / dst_height;
  for (int i = 0; i < dst_size; ++i) {
    // destination position
    int dst_x = i % dst_width;
    int dst_y = i / dst_width;
    int dst_index = dst_y * dst_width + dst_x;

    // source position
    int left_x = floor(dst_x * x_ratio);
    int right_x = floor((dst_x + 1) * x_ratio);
    int top_y = floor(dst_y * y_ratio);
    int bottom_y = floor((dst_y + 1) * y_ratio);
    int area_size = (right_x - left_x) * (bottom_y - top_y);

    int sum_of_pixels = 0;
    for (int y = top_y; y < bottom_y; ++y)
      for (int x = left_x; x < right_x; ++x)
        sum_of_pixels += src.at(y * src_width + x);
    dst->data()[dst_index] = (uint8_t)(sum_of_pixels / area_size);
  }
}

};
