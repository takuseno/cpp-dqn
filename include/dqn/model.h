#pragma once

#include <nbla/functions.hpp>
#include <nbla/parametric_functions.hpp>
#include <nbla/solver/rmsprop.hpp>
#include <nbla/computation_graph/computation_graph.hpp>

using std::make_shared;
using namespace nbla;
namespace f = nbla::functions;
namespace pf = nbla::parametric_functions;

namespace dqn {

class Model
{
public:
  Model(int num_of_actions, int batch_size, float gamma, float lr, Context ctx);
  void infer(const uint8_t* obs_t, float* q_values);
  float train(const uint8_t* obss_t, const uint8_t* acts_t, const float* rews_tp1,
              const uint8_t* obss_tp1, const float* ters_tp1);
  void sync_target();

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
  void set_image(CgVariablePtr x, const uint8_t* image, int batch_size);
  template <typename T>
  void set_data(CgVariablePtr x, const T* data, int batch_size);
};

}
