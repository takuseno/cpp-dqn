cmake_minimum_required(VERSION 3.1)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(dqn CXX)

option(USE_SDL "Use SDL for ALE" ON)
option(DEBUG "Enable debug option" OFF)

if (DEBUG)
  set(COMPILE_OPTION "-g -O0")
else()
  set(COMPILE_OPTION "-O3")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ${COMPILE_OPTION}")

add_subdirectory(third_party)
add_subdirectory(src/dqn)
add_subdirectory(tests)
add_subdirectory(examples)
