#pragma once

#include <dqn/models/controller.h>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

#define LOG_BASE_DIR "logs"

using namespace std;
using namespace nbla;

namespace dqn {

class Monitor {
public:
  Monitor(const string &logdir);
  void print(const string &name, int t, float value);
  void print(const string &name, int t, const vector<float> &values);
  void add(const string &name);
  void close(const string &name);
  void save_parameters(int t, shared_ptr<Controller> controller);

private:
  void prepare_directory();
  unordered_map<string, FILE *> fps_;
  string logdir_;
};

class MonitorSeries {
public:
  MonitorSeries(shared_ptr<Monitor> monitor, const string &name, int window);
  ~MonitorSeries() { monitor_->close(name_); };
  void add(float value);
  void emit(int t);

private:
  shared_ptr<Monitor> monitor_;
  string name_;
  int window_;
  int count_;
  vector<float> history_;
};

class MonitorMultiColumnSeries {
public:
  MonitorMultiColumnSeries(shared_ptr<Monitor> monitor, const string &name,
                           int num_columns);
  ~MonitorMultiColumnSeries() { monitor_->close(name_); };
  void add(float value);
  void emit(int t);

private:
  shared_ptr<Monitor> monitor_;
  string name_;
  int num_columns_, cursor_;
  vector<float> history_;
};

}; // namespace dqn
