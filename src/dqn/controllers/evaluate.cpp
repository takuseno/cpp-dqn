#include <dqn/controllers/evaluate.h>

namespace dqn {

EvaluateController::EvaluateController(shared_ptr<Model> model,
                                       shared_ptr<Exploration> exploration) {
  model_ = model;
  exploration_ = exploration;
}

uint8_t EvaluateController::act(int t, const vector<uint8_t> &obs_t) {
  vector<float> q_values;
  model_->infer(obs_t, &q_values);
  return exploration_->sample(q_values.data(), t);
}

}; // namespace dqn
