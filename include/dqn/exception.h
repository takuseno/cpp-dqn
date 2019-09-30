#pragma once

#include <exception>
#include <iostream>
#include <sstream>
#include <string>

#define _FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define DQN_CHECK(cond, message)                                               \
  if (!(cond)) {                                                               \
    throw Exception(#cond, message, _FILE, __LINE__ - 1);                      \
  }

using namespace std;

namespace dqn {

class Exception : public exception {
public:
  Exception(string cond, string message, string file, int line);
  virtual ~Exception() throw() {}
  virtual const char *what() const throw() { return full_message_.c_str(); }

private:
  string cond_;
  string message_;
  string file_;
  string full_message_;
  int line_;
};

}; // namespace dqn
