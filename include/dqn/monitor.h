#pragma once

#include <stdio.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <numeric>

using namespace std;

namespace dqn {

class Monitor {
public:
  Monitor(const string& logdir);
  void print(const string& name, int t, float value);
  void add(const string& name);
  void close(const string& name);

private:
  void prepare_directory();
  unordered_map<string, FILE*> fps_;
  string logdir_;
};

class MonitorSeries {
public:
  MonitorSeries(shared_ptr<Monitor> monitor, const string& name, int interval);
  ~MonitorSeries() { monitor_->close(name_); };
  void add(int t, float value);

private:
  shared_ptr<Monitor> monitor_;
  string name_;
  int interval_;
  int count_;
  vector<float> history_;
};

};
