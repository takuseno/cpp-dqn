#include <dqn/monitor.h>

namespace dqn {

Monitor::Monitor(const string& logdir) {
  logdir_ = logdir;
}

void Monitor::add(const string& name) {
  FILE* fp = fopen((logdir_ + "/" + name).c_str(), "wt");
  fps_[name] = fp;
}

void Monitor::close(const string& name) {
  fclose(fps_[name]);
}

void Monitor::print(const string& name, int t, float value) {
  fprintf(fps_[name], "%d %f\n", t, value);
}

MonitorSeries::MonitorSeries(shared_ptr<Monitor> monitor, const string& name, int interval) {
  monitor_ = monitor;
  name_ = name;
  interval_ = interval;
  history_.resize(interval);
  count_ = 0;
}

void MonitorSeries::add(int t, float value) {
  history_[count_ % interval_] = value;
  ++count_;

  if (count_ % interval_ != 0)
    return;

  float sum = accumulate(begin(history_), end(history_), 0.0);
  float mean = sum / interval_;

  fprintf(stdout, "[%s] step: %d, value %f\n", name_.c_str(), t, mean);

  if (monitor_ == nullptr)
    return;
  monitor_->print(name_, t, mean);
}

};
