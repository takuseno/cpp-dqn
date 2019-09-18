cmake_minimum_required(VERSION 3.1)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
project(dqn CXX)

option(USE_SDL "Use SDL for ALE" ON)

add_subdirectory(third_party)
add_subdirectory(src/dqn)
add_subdirectory(tests)
add_subdirectory(examples)
