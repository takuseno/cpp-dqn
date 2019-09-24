add_executable(run_game run_game.cpp)
target_link_libraries(run_game dqn)

add_executable(train train.cpp)
target_link_libraries(train dqn)
