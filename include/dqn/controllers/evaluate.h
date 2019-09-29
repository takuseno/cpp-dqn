#pragma once

#include <dqn/controllers/controller.h>
#include <dqn/explorations/exploration.h>
#include <dqn/models/model.h>

using namespace std;

namespace dqn {

class EvaluateController : public Controller {
public:
  EvaluateController(shared_ptr<Model> model,
                     shared_ptr<Exploration> exploration);
  virtual uint8_t act(int t, const vector<uint8_t> &obs_t);
  virtual void store(int t, const vector<uint8_t> &obs_t, uint8_t act_t,
                     float rew_tp1, const vector<uint8_t> &obs_tp1,
                     float ter_tp1) {}
  virtual float update(int t) { return -1; }
  virtual void load(const char *path) {}
  virtual void save(const char *path) {}

private:
  shared_ptr<Model> model_;
  shared_ptr<Exploration> exploration_;
};

}; // namespace dqn
