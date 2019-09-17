enable_testing()

file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  ./*.cpp)

add_executable(dqntest ${SOURCES})

include_directories(${PROJECT_SOURCE_DIR}/third_party/googletest-release-1.7.0/include)

find_library(NNABLA_LIBRARY nnabla HINTS /usr/local/lib)

target_link_libraries(dqntest ${NNABLA_LIBRARY} dqn gtest gtest_main)

add_test(NAME dqntest COMMAND dqntest)
