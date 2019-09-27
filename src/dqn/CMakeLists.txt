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

find_library(NNABLA_LIBRARY nnabla HINTS /usr/local/lib)
target_link_libraries(dqn
  PUBLIC
  ${NNABLA_LIBRARY}
  ${PROJECT_SOURCE_DIR}/third_party/Arcade-Learning-Environment-master/libale.so)
