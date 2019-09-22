#pragma once

#include <nbla/computation_graph/computation_graph.hpp>
#include <vector>

using namespace std;
using namespace nbla;

namespace dqn {

inline void set_image(CgVariablePtr x, const vector<vector<uint8_t> *> &image,
                      Context ctx) {
  uint8_t *x_d = x->variable()->cast_data_and_get_pointer<uint8_t>(ctx, true);
  for (int i = 0; i < image.size(); ++i) {
    int offset = i * image[i]->size();
    memcpy(x_d + offset, image[i]->data(), image[i]->size());
  }
}

template <typename T>
void set_data(CgVariablePtr x, const vector<T> &data, Context ctx) {
  T *x_d = x->variable()->cast_data_and_get_pointer<T>(ctx, true);
  memcpy(x_d, data.data(), sizeof(T) * data.size());
}

}; // namespace dqn
