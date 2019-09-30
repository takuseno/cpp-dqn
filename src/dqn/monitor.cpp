#include <dqn/monitor.h>

namespace dqn {

Monitor::Monitor(const string &logdir) {
  logdir_ = logdir;
  prepare_directory();
}

void Monitor::add(const string &name) {
  string base_path = LOG_BASE_DIR;
  string path = base_path + "/" + logdir_ + "/" + name + ".csv";
  FILE *fp;
  if ((fp = fopen(path.c_str(), "wt")) == NULL) {
    fprintf(stderr, "failed to create %s\n", path.c_str());
    exit(1);
  }
  fps_[name] = fp;
}

void Monitor::close(const string &name) { fclose(fps_[name]); }

void Monitor::print(const string &name, int t, float value) {
  fprintf(fps_[name], "%d,%f\n", t, value);
  fflush(fps_[name]);
}

void Monitor::print(const string &name, int t, const vector<float> &values) {
  ostringstream ss;
  ss << t;
  for (int i = 0; i < values.size(); ++i)
    ss << "," << values[i];
  fprintf(fps_[name], "%s\n", ss.str().c_str());
  fflush(fps_[name]);
}

void Monitor::prepare_directory() {
  string base_path = LOG_BASE_DIR;
  string path = base_path + "/" + logdir_;
  if (mkdir(path.c_str(), 0755)) {
    fprintf(stderr, "failed to create directory at %s\n", path.c_str());
    exit(1);
  }
}

void Monitor::save_parameters(int t, shared_ptr<Controller> controller) {
  ostringstream ss;
  ss << LOG_BASE_DIR << "/" << logdir_ << "/"
     << "model_" << t << ".nnp";
  controller->save(ss.str().c_str());
}

MonitorSeries::MonitorSeries(shared_ptr<Monitor> monitor, const string &name,
                             int window) {
  monitor_ = monitor;
  name_ = name;
  window_ = window;
  history_.resize(window, 0);
  count_ = 0;

  if (monitor != nullptr) {
    monitor_->add(name_);
  }
}

void MonitorSeries::add(float value) {
  history_[count_ % window_] = value;
  ++count_;
}

void MonitorSeries::emit(int t) {
  if (count_ == 0)
    return;

  float sum = accumulate(begin(history_), end(history_), 0.0);
  int n = count_ > window_ ? window_ : count_;
  float mean = sum / n;

  fprintf(stdout, "[%s] step: %d, value %f\n", name_.c_str(), t, mean);

  if (monitor_ == nullptr)
    return;
  monitor_->print(name_, t, mean);
}

MonitorMultiColumnSeries::MonitorMultiColumnSeries(shared_ptr<Monitor> monitor,
                                                   const string &name,
                                                   int num_columns) {
  monitor_ = monitor;
  name_ = name;
  num_columns_ = num_columns;
  history_.resize(num_columns, 0.0);
  cursor_ = 0;

  if (monitor != nullptr)
    monitor_->add(name_);
}

void MonitorMultiColumnSeries::add(float value) {
  history_[cursor_] = value;
  ++cursor_;
  if (cursor_ == num_columns_)
    cursor_ = 0;
}

void MonitorMultiColumnSeries::emit(int t) {
  float sum = accumulate(begin(history_), end(history_), 0.0);
  float mean = sum / history_.size();

  fprintf(stdout, "[%s] step: %d, mean value %f\n", name_.c_str(), t, mean);

  if (monitor_ != nullptr)
    monitor_->print(name_, t, history_);

  cursor_ = 0;
}

}; // namespace dqn
