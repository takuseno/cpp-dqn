#pragma once

#include <dqn/buffer.h>
#include <dqn/encoder.h>
#include <dqn/models/model.h>
#include <dqn/nnabla_utils.h>
#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/functions.hpp>
#include <nbla/parametric_functions.hpp>
#include <nbla/solver/rmsprop.hpp>

using namespace std;
using namespace nbla;
namespace f = nbla::functions;
namespace pf = nbla::parametric_functions;

namespace dqn {

class DQN : public Model {
public:
  DQN(int num_of_actions, int batch_size, float gamma, float lr, Context ctx);
  virtual void infer(const vector<uint8_t> &obs_t, vector<float> *q_values);
  virtual float train(BatchPtr batch);
  virtual void sync_target();

protected:
  virtual void build();
  virtual CgVariablePtr q_network(CgVariablePtr obss_t,
                                  ParameterDirectory params);

private:
  float gamma_, lr_;
  shared_ptr<Solver> solver_;
  // inputs
  CgVariablePtr obs_t_, obss_t_, acts_t_, rews_tp1_, obss_tp1_, ters_tp1_;
  // outputs
  CgVariablePtr q_values_, loss_;
  vector<CgVariablePtr> assigns_;
};

}; // namespace dqn
