#!/bin/bash

clang-format -i include/**/*.h
clang-format -i src/**/*.cpp
clang-format -i tests/*.cpp
clang-format -i examples/*.cpp
