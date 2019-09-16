enable_testing()

file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  ./*.cpp)

add_executable(dqntest ${SOURCES})
include_directories(${PROJECT_SOURCE_DIR}/third_party/googletest-release-1.7.0/include)
target_link_libraries(dqntest dqn gtest gtest_main)


add_test(NAME dqntest COMMAND dqntest)
