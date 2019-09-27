#include "dqn_test_utils.h"
#include "gtest/gtest.h"
#include <dqn/models/dqn.h>
#include <nbla/computation_graph/computation_graph.hpp>
#include <random>

bool check_same(CgVariablePtr x, CgVariablePtr y, nbla::Context ctx) {
  float_t *x_d = x->variable()->cast_data_and_get_pointer<float_t>(ctx, true);
  float_t *y_d = y->variable()->cast_data_and_get_pointer<float_t>(ctx, true);
  if (x->variable()->size() != y->variable()->size())
    return false;
  for (int i = 0; i < x->variable()->size(); ++i)
    if (x_d[i] != y_d[i])
      return false;
  return true;
}

TEST(ModelTest, ConstructGraph) {
  nbla::Context ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  dqn::DQN model(4, 32, 0.99, 0.00025, ctx);

  vector<uint8_t> obs_t(4 * 84 * 84);
  fill_vector(&obs_t);
  vector<uint8_t> obs_tp1(4 * 84 * 84);
  fill_vector(&obs_tp1);

  // check inference
  vector<float> q_values;
  model.infer(obs_t, &q_values);
  for (int i = 0; i < 4; ++i) {
    ASSERT_NE(q_values[i], 0);
  }

  // check sync_target
  ParameterDirectory params = model.parameter_directory();
  auto trainable_params = params["trainable"].get_parameters();
  auto target_params = params["target"].get_parameters();
  model.sync_target();
  for (int i = 0; i < trainable_params.size(); ++i) {
    auto x = make_shared<CgVariable>(target_params[i].second);
    auto y = make_shared<CgVariable>(trainable_params[i].second);
    ASSERT_EQ(check_same(x, y, ctx), true);
  }

  // check train
  dqn::Batch batch;
  batch.obss_t.resize(32);
  batch.acts_t.resize(32);
  batch.obss_tp1.resize(32);
  batch.rews_tp1.resize(32);
  batch.ters_tp1.resize(32);
  for (int i = 0; i < 32; ++i) {
    batch.obss_t[i] = &obs_t;
    batch.acts_t[i] = 0;
    batch.obss_tp1[i] = &obs_tp1;
    batch.rews_tp1[i] = 0;
    batch.ters_tp1[i] = 0;
  }

  float loss = model.train(make_shared<dqn::Batch>(batch));
  ASSERT_NE(loss, 0);

  for (int i = 0; i < trainable_params.size(); ++i) {
    auto x = make_shared<CgVariable>(target_params[i].second);
    auto y = make_shared<CgVariable>(trainable_params[i].second);
    ASSERT_EQ(check_same(x, y, ctx), false);
  }
}
