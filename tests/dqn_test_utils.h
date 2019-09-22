#pragma once

#include <vector>
#include <random>

inline void fill_vector(std::vector<uint8_t> *v) {
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  std::uniform_int_distribution<> dist(0, 255);
  for (int i = 0; i < v->size(); ++i)
    (*v)[i] = dist(engine);
}

inline void fill_vector(std::vector<float> *v) {
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  std::uniform_real_distribution<> dist(0.0, 1.0);
  for (int i = 0; i < v->size(); ++i)
    (*v)[i] = dist(engine);
}
