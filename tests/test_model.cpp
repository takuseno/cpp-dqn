#include "gtest/gtest.h"
#include <dqn/model.h>
#include <nbla/computation_graph/computation_graph.hpp>

TEST(ModelTest, ConstructGraph) {
  nbla::Context ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  dqn::Model model(4, 32, 0.99, 0.00025, ctx);
}
