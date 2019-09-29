#pragma once

#include <dqn/buffer.h>
#include <dqn/controllers/controller.h>
#include <dqn/explorations/exploration.h>
#include <dqn/models/model.h>

using namespace std;

namespace dqn {

class DQNController : public Controller {
public:
  DQNController(shared_ptr<Model> model, shared_ptr<Buffer> buffer,
                shared_ptr<Exploration> exploration, int update_start,
                int update_interval, int target_update_interval);
  virtual uint8_t act(int t, const vector<uint8_t> &obs_t);
  virtual void store(int t, const vector<uint8_t> &obs_t, uint8_t act_t,
                     float rew_tp1, const vector<uint8_t> &obs_tp1,
                     float ter_tp1);
  virtual float update(int t);
  virtual void save(const char *path) { model_->save(path); }
  virtual void load(const char *path) { model_->load(path); }

private:
  int update_start_, update_interval_, target_update_interval_;
  shared_ptr<Model> model_;
  shared_ptr<Buffer> buffer_;
  shared_ptr<Exploration> exploration_;
};

}; // namespace dqn
