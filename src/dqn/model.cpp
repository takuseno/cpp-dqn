#include <dqn/model.h>

namespace dqn {

Model::Model(int num_of_actions, int batch_size, float gamma, float lr,
             Context ctx) {
  num_of_actions_ = num_of_actions;
  batch_size_ = batch_size;
  gamma_ = gamma;
  lr_ = lr;
  ctx_ = ctx;
  cpu_ctx_ = Context({"cpu:float"}, "CpuCachedArray", "0");
  build();
}

void Model::build() {
  params_ = ParameterDirectory();

  // entry variables
  obs_t_ = make_shared<CgVariable>(Shape_t({1, 4, 84, 84}), false);
  obss_t_ = make_shared<CgVariable>(Shape_t({batch_size_, 4, 84, 84}), false);
  acts_t_ = make_shared<CgVariable>(Shape_t({batch_size_, 1}), false);
  rews_tp1_ = make_shared<CgVariable>(Shape_t({batch_size_, 1}), false);
  obss_tp1_ = make_shared<CgVariable>(Shape_t({batch_size_, 4, 84, 84}), false);
  ters_tp1_ = make_shared<CgVariable>(Shape_t({batch_size_, 1}), false);

  // inference
  q_values_ = q_network(obs_t_ / 255.0, params_["trainable"]);

  // training
  auto q_t = q_network(obss_t_ / 255.0, params_["trainable"]);
  auto q_tp1 = q_network(obss_tp1_ / 255.0, params_["target"]);

  auto a_one_hot = f::one_hot(acts_t_, {num_of_actions_});
  auto q_t_selected = f::sum(q_t * a_one_hot, {1}, true);
  auto q_tp1_best = f::max(q_tp1, {1}, true, false, false);

  // target value
  auto y = rews_tp1_ + gamma_ * q_tp1_best * (1.0 - ters_tp1_);
  // loss
  loss_ = f::mean(f::huber_loss(q_t_selected, y, 1.0), {}, false);

  // target update
  auto trainable_params = params_["trainable"].get_parameters();
  auto target_params = params_["target"].get_parameters();
  for (int i = 0; i < trainable_params.size(); ++i) {
    auto dst = make_shared<CgVariable>(target_params[i].second);
    auto src = make_shared<CgVariable>(trainable_params[i].second);
    auto assign = f::assign(dst, src);
    assigns_.push_back(assign);
  }

  // setup solver
  solver_ = create_RMSpropSolver(ctx_, lr_, 0.95, 0.01);
  solver_->set_parameters(params_.get_parameters());
}

void Model::infer(const vector<uint8_t> &obs_t, float *q_values) {
  vector<vector<uint8_t> *> v_obs_t = {(vector<uint8_t> *)&obs_t};
  set_image(obs_t_, v_obs_t, cpu_ctx_);
  q_values_->forward(true, true);
  float_t *q_values_d =
      q_values_->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx_,
                                                                false);
  memcpy(q_values, q_values_d, sizeof(float) * num_of_actions_);
}

float Model::train(BatchPtr batch) {
  set_image(obss_t_, batch->obss_t, cpu_ctx_);
  set_data(acts_t_, batch->acts_t, cpu_ctx_);
  set_data(rews_tp1_, batch->rews_tp1, cpu_ctx_);
  set_image(obss_tp1_, batch->obss_tp1, cpu_ctx_);
  set_data(ters_tp1_, batch->ters_tp1, cpu_ctx_);

  solver_->zero_grad();
  loss_->forward(false, true);
  loss_->variable()->grad()->fill(1.0);
  loss_->backward(nullptr, true);
  solver_->update();

  float_t *loss_d =
      loss_->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx_, false);
  return loss_d[0];
}

void Model::sync_target() {
  for (int i = 0; i < assigns_.size(); ++i) {
    assigns_[i]->forward(true, true);
  }
}

CgVariablePtr Model::q_network(CgVariablePtr obss_t,
                               ParameterDirectory params) {
  pf::ConvolutionOpts opts1 = pf::ConvolutionOpts().stride({4, 4});
  auto h = pf::convolution(obss_t, 1, 32, {8, 8}, params["conv1"], opts1);
  h = f::relu(h, true);
  pf::ConvolutionOpts opts2 = pf::ConvolutionOpts().stride({2, 2});
  h = pf::convolution(h, 1, 64, {4, 4}, params["conv2"], opts2);
  h = f::relu(h, true);
  pf::ConvolutionOpts opts3 = pf::ConvolutionOpts().stride({1, 1});
  h = pf::convolution(h, 1, 64, {3, 3}, params["conv3"], opts3);
  h = f::relu(h, true);
  h = pf::affine(h, 1, 512, params["fc1"]);
  h = f::relu(h, true);
  h = pf::affine(h, 1, num_of_actions_, params["fc2"]);
  return h;
}

}; // namespace dqn
