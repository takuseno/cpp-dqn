#pragma once

#include <dqn/buffer.h>
#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/parametric_functions.hpp>

using namespace std;
using namespace nbla;

namespace dqn {

class Model {
public:
  Model(int num_of_actions, int batch_size, Context ctx) {
    num_of_actions_ = num_of_actions;
    batch_size_ = batch_size;
    params_ = ParameterDirectory();
    ctx_ = ctx;
    cpu_ctx_ = Context({"cpu:float"}, "CpuCachedArray", "0");
  }
  virtual ~Model() {}
  virtual void infer(const vector<uint8_t> &obs_t, vector<float> *q_values) = 0;
  virtual float train(BatchPtr batch) = 0;
  virtual void sync_target() = 0;
  virtual ParameterDirectory parameter_directory() { return params_; };
  virtual int batch_size() { return batch_size_; }

protected:
  int num_of_actions_, batch_size_;
  ParameterDirectory params_;
  Context ctx_, cpu_ctx_;
};

}; // namespace dqn
