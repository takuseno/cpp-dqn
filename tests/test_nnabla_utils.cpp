#include "dqn_test_utils.h"
#include "gtest/gtest.h"
#include <dqn/nnabla_utils.h>

TEST(NNablaUtilsTest, SetImage) {
  vector<vector<uint8_t> *> batch;
  vector<uint8_t> v1(4 * 84 * 84);
  vector<uint8_t> v2(4 * 84 * 84);
  fill_vector(&v1);
  fill_vector(&v2);
  batch.push_back(&v1);
  batch.push_back(&v2);

  nbla::Context ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  auto x = make_shared<nbla::CgVariable>(nbla::Shape_t({2, 4, 84, 84}), false);

  dqn::set_image(x, batch, ctx);

  uint8_t *x_d = x->variable()->cast_data_and_get_pointer<uint8_t>(ctx, true);
  for (int i = 0; i < v1.size(); ++i) {
    ASSERT_EQ(x_d[i], v1[i]);
  }
  for (int i = 0; i < v2.size(); ++i) {
    ASSERT_EQ(x_d[i + v1.size()], v2[i]);
  }
}

TEST(NNablaUtilsTest, SetData) {
  vector<float> v(32);
  fill_vector(&v);

  nbla::Context ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  auto x = make_shared<nbla::CgVariable>(nbla::Shape_t({32, 1}), false);

  dqn::set_data(x, v, ctx);

  float *x_d = x->variable()->cast_data_and_get_pointer<float>(ctx, true);
  for (int i = 0; i < v.size(); ++i)
    ASSERT_EQ(x_d[i], v[i]);
}
