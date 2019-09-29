file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  ./*.cpp
  ./models/*.cpp
  ./controllers/*.cpp
  ./explorations/*.cpp)

add_library(dqn SHARED ${SOURCES})

target_include_directories(dqn PUBLIC ${PROJECT_SOURCE_DIR}/third_party/Arcade-Learning-Environment-master/src)
target_include_directories(dqn PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(dqn PUBLIC /usr/local/include)

# SDL
if (USE_SDL)
  find_library(SDL_LIBRARY SDL HINTS /usr/local/lib)
  target_link_libraries(dqn PUBLIC ${SDL_LIBRARY})
  find_library(SDL_GFX_LIBRARY SDL_gfx HINTS /usr/local/lib)
  target_link_libraries(dqn PUBLIC ${SDL_GFX_LIBRARY})
  find_library(SDL_IMAGE_LIBRARY SDL_image HINTS /usr/local/lib)
  target_link_libraries(dqn PUBLIC ${SDL_IMAGE_LIBRARY})
  find_library(SDL_MAIN_LIBRARY SDLmain HINTS /usr/local/lib)
  target_link_libraries(dqn PUBLIC ${SDL_MAIN_LIBRARY})
endif()

# ALE
target_link_libraries(dqn PUBLIC ${PROJECT_SOURCE_DIR}/third_party/Arcade-Learning-Environment-master/libale.so)

# NNabla
if(APPLE)
  target_include_directories(dqn PUBLIC ${PROJECT_SOURCE_DIR}/third_party/nnabla/include)
  target_link_libraries(dqn PUBLIC ${PROJECT_SOURCE_DIR}/third_party/nnabla/cpu/lib/libnnabla.dylib)
  target_link_libraries(dqn PUBLIC ${PROJECT_SOURCE_DIR}/third_party/nnabla/cpu/lib/libnnabla_utils.dylib)
else()
  find_library(NNABLA_LIBRARY nnabla HINTS /usr/local/lib)
  find_library(NNABLA_UTILS_LIBRARY nnabla_utils HINTS /usr/local/lib)
  target_link_libraries(dqn PUBLIC ${NNABLA_LIBRARY} ${NNABLA_UTILS_LIBRARY})
endif()

# NNabla CUDA extension
if (GPU)
  find_library(NNABLA_CUDA_LIBRARY nnabla_cuda HINTS /usr/local/lib)
  target_link_libraries(dqn PUBLIC ${NNABLA_CUDA_LIBRARY})
endif()
