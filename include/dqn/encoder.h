#pragma once

#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/functions.hpp>
#include <nbla/parametric_functions.hpp>

using namespace nbla;
namespace f = nbla::functions;
namespace pf = nbla::parametric_functions;

namespace dqn {

inline CgVariablePtr nature_encoder(CgVariablePtr x,
                                    ParameterDirectory params) {
  pf::ConvolutionOpts opts1 = pf::ConvolutionOpts().stride({4, 4});
  auto h = pf::convolution(x, 1, 32, {8, 8}, params["conv1"], opts1);
  h = f::relu(h, false);
  pf::ConvolutionOpts opts2 = pf::ConvolutionOpts().stride({2, 2});
  h = pf::convolution(h, 1, 64, {4, 4}, params["conv2"], opts2);
  h = f::relu(h, false);
  pf::ConvolutionOpts opts3 = pf::ConvolutionOpts().stride({1, 1});
  h = pf::convolution(h, 1, 64, {3, 3}, params["conv3"], opts3);
  h = f::relu(h, false);
  h = pf::affine(h, 1, 512, params["fc1"]);
  h = f::relu(h, false);
  return h;
}

}; // namespace dqn
