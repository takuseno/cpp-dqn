function(download_and_extract_library NAME EXT URL)
  if(NOT IS_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/${NAME})
    message("Downloading ${NAME}")
    file(DOWNLOAD
      ${URL}
      ${CMAKE_SOURCE_DIR}/third_party/${NAME}${EXT}
      SHOW_PROGRESS
      STATUS status
      LOG log)
    list(GET status 0 status_code)
    list(GET status 1 status_string)
    if(NOT status_code EQUAL 0)
      if(UNIX OR APPLE)
        message("Failed to download ${NAME}. Try to download with wget.")
        execute_process(
          COMMAND wget ${URL} -O ${CMAKE_SOURCE_DIR}/third_party/${NAME}${EXT})
      else()
        message(FATAL_ERROR "Failed to download ${NAME}. Please check network connection and proxy settings.
        status_code: ${status_code}
        status_string: ${status_string}
        log: ${log}
        ")
        error() # Place Unknown command to stop cmake process here.
      endif()

    endif()
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E
      chdir ${CMAKE_SOURCE_DIR}/third_party
      ${CMAKE_COMMAND} -E tar xf ${NAME}${EXT})
  endif()
endfunction()

# google test
download_and_extract_library(googletest-release-1.7.0 .zip https://github.com/google/googletest/archive/release-1.7.0.zip)
add_subdirectory(googletest-release-1.7.0)

# google flags
download_and_extract_library(gflags-2.2.2 .zip https://github.com/gflags/gflags/archive/v2.2.2.zip)
add_subdirectory(gflags-2.2.2)

# ALE
if(USE_SDL)
  add_definitions(-D__USE_SDL)
endif()
download_and_extract_library(Arcade-Learning-Environment-master .zip https://github.com/mgbellemare/Arcade-Learning-Environment/archive/master.zip)
add_subdirectory(Arcade-Learning-Environment-master)

if(APPLE)
  if(NOT IS_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/nnabla)
    download_and_extract_library(nnabla_bin_macos_v1.1.0 .zip https://github.com/sony/nnabla/releases/download/v1.1.0/nnabla_bin_macos_v1.1.0.zip)
    execute_process(COMMAND ${CMAKE_COMMAND} -E
      chdir ${CMAKE_SOURCE_DIR}/third_party
      mkdir nnabla)
    execute_process(COMMAND ${CMAKE_COMMAND} -E
      chdir ${CMAKE_SOURCE_DIR}/third_party
      mv include nnabla/)
    execute_process(COMMAND ${CMAKE_COMMAND} -E
      chdir ${CMAKE_SOURCE_DIR}/third_party
      mv cpu nnabla/)
  endif()
endif()

# Download ROMs
if(NOT IS_DIRECTORY ${CMAKE_SOURCE_DIR}/atari_roms)
  message("Downloading ROMs")
  file(DOWNLOAD
      https://github.com/openai/atari-py/archive/0.2.6.zip
      ${CMAKE_SOURCE_DIR}/third_party/0.2.6.zip
      SHOW_PROGRESS)
  execute_process(COMMAND ${CMAKE_COMMAND} -E
    chdir ${CMAKE_SOURCE_DIR}/third_party
    unzip 0.2.6.zip)
  execute_process(COMMAND ${CMAKE_COMMAND} -E
    chdir ${CMAKE_SOURCE_DIR}/third_party/atari-py-0.2.6
    mv atari_py/atari_roms ${CMAKE_SOURCE_DIR}/atari_roms)
endif()
