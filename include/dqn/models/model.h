#pragma once

#include <dqn/buffer.h>
#include <dqn/exception.h>
#include <fstream>
#include <iostream>
#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/parametric_functions.hpp>
using namespace nbla;

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
  virtual void save(const char *path) {
    // TODO: replace this process with official NNabla API
    auto params = params_.get_parameters();
    int num_of_variables = params.size();
    vector<int> sizes;
    for (int i = 0; i < num_of_variables; ++i)
      sizes.push_back(params[i].second->size());
    ofstream fout;
    fout.open(path, ios::out | ios::binary | ios::trunc);
    DQN_CHECK(fout, "failed to create file");

    // header
    fout.write((char *)&num_of_variables, sizeof(int));
    for (int i = 0; i < num_of_variables; ++i)
      fout.write((char *)&sizes[i], sizeof(int));

    // parameters
    for (int i = 0; i < num_of_variables; ++i) {
      auto variable = params[i].second;
      const float *v_d = variable->get_data_pointer<float>(cpu_ctx_);
      fout.write((char *)v_d, sizeof(float) * variable->size());
    }

    fout.close();
  }
  virtual void load(const char *path) {
    // TODO: replace this process with official NNabla API
    auto params = params_.get_parameters();

    ifstream fin(path, ios::in | ios::binary);
    DQN_CHECK(fin, "failed to open file");

    int num_of_variables;
    fin.read((char *)&num_of_variables, sizeof(int));
    DQN_CHECK(num_of_variables == params.size(),
              "mismatched number of variables");

    vector<int> sizes;
    for (int i = 0; i < num_of_variables; ++i) {
      int size;
      fin.read((char *)&size, sizeof(int));
      DQN_CHECK(size == params[i].second->size(), "mismatched variable size");
      sizes.push_back(size);
    }

    for (int i = 0; i < num_of_variables; ++i) {
      auto variable = params[i].second;
      float *v_d = variable->cast_data_and_get_pointer<float>(cpu_ctx_, true);
      fin.read((char *)v_d, sizeof(float) * sizes[i]);
    }

    fin.close();
  }

protected:
  int num_of_actions_, batch_size_;
  ParameterDirectory params_;
  Context ctx_, cpu_ctx_;
};

}; // namespace dqn
