#pragma once

#include <algorithm>
#include <array>
#include <dqn/exception.h>
#include <vector>

namespace dqn {

void resize(vector<uint8_t> *dst, const vector<uint8_t> &src,
            const array<int, 2> &dst_shape, const array<int, 2> &src_shape);

}; // namespace dqn
