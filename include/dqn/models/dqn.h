#pragma once

#include <dqn/buffer.h>
#include <dqn/nnabla_utils.h>
#include <dqn/encoder.h>
#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/functions.hpp>
#include <nbla/parametric_functions.hpp>
#include <nbla/solver/rmsprop.hpp>

using namespace std;
using namespace nbla;
namespace f = nbla::functions;
namespace pf = nbla::parametric_functions;

namespace dqn {

class DQN {
public:
  DQN(int num_of_actions, int batch_size, float gamma, float lr, Context ctx);
  void infer(const vector<uint8_t> &obs_t, float *q_values);
  float train(BatchPtr batch);
  void sync_target();
  ParameterDirectory parameter_directory() { return params_; };
  int batch_size() { return batch_size_; }

private:
  int num_of_actions_, batch_size_;
  float gamma_, lr_;
  Context ctx_, cpu_ctx_;
  ParameterDirectory params_;
  shared_ptr<Solver> solver_;
  // inputs
  CgVariablePtr obs_t_, obss_t_, acts_t_, rews_tp1_, obss_tp1_, ters_tp1_;
  // outputs
  CgVariablePtr q_values_, loss_;
  vector<CgVariablePtr> assigns_;

  void build();
  CgVariablePtr q_network(CgVariablePtr obss_t, ParameterDirectory params);
};

}; // namespace dqn
