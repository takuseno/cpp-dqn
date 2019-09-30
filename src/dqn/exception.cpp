#include <dqn/exception.h>

namespace dqn {

Exception::Exception(string cond, string message, string file, int line) {
  cond_ = cond;
  message_ = message;
  file_ = file;
  line_ = line;
  ostringstream ss;
  ss << cond_ << " check failed at " << line_ << " of " << file_ << endl
     << message_ << endl;
  full_message_ = ss.str();
}

}; // namespace dqn
