add_executable(run_game run_game.cpp)
target_link_libraries(run_game dqn)

include_directories(${PROJECT_BINARY_DIR}/third_party/gflags-2.2.2/include)
add_executable(train train.cpp)
target_link_libraries(train dqn gflags)
