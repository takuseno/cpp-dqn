file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  ./*.cpp)

add_library(dqn SHARED ${SOURCES})

target_include_directories(dqn PUBLIC ${PROJECT_SOURCE_DIR}/third_party/Arcade-Learning-Environment-master/src)
target_include_directories(dqn PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(dqn PUBLIC /usr/local/include)

find_package(ZLIB REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(dqn
  /usr/local/lib/libnnabla.dylib
  ${PROJECT_SOURCE_DIR}/third_party/Arcade-Learning-Environment-master/libale.so
  ${OpenCV_LIBS}
  ${ZLIB_LIBRARIES})
