#pragma once

#include <stdio.h>

#define DQN_CHECK(cond, message) \
  if (cond) {\
    fprintf(stderr, "%s: %s at %d of %s\n", "#cond", message, __LINE__, \
            __FILE__); \
    exit(1); \
  }
