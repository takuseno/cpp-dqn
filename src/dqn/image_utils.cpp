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
    float left_x = (float)dst_x * x_ratio;
    float right_x = std::min((float)(dst_x + 1) * x_ratio, (float)src_width);
    float top_y = (float)dst_y * y_ratio;
    float bottom_y = std::min((float)(dst_y + 1) * y_ratio, (float)src_height);
    float area_size = (right_x - left_x) * (bottom_y - top_y);

    // accumulate pixels based on kernel map
    float sum_of_pixels = 0.0;
    for (int y = top_y; y <= (int)bottom_y; ++y) {
      // calculate weight of y-axis
      // reduce weights for extra pixels
      float y_weight = 1.0;
      if (y == (int)top_y)
        y_weight = y + 1.0 - top_y;
      else if (y == (int)bottom_y)
        y_weight = bottom_y - y;
      for (int x = left_x; x <= (int)right_x; ++x) {
        // calculate weight of x-axis
        float x_weight = 1.0;
        if (x == (int)left_x)
          x_weight = x + 1.0 - left_x;
        else if (x == (int)right_x)
          x_weight = right_x - x;
        float weight = x_weight * y_weight;
        if (weight > 0)
          sum_of_pixels += weight * (float)src.at(y * src_width + x);
      }
    }
    dst->data()[dst_index] = sum_of_pixels / area_size;
  }
}

}; // namespace dqn
