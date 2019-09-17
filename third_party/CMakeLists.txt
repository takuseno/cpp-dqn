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

# ALE
if(USE_SDL)
  add_definitions(-D__USE_SDL)
endif()
download_and_extract_library(Arcade-Learning-Environment-master .zip https://github.com/mgbellemare/Arcade-Learning-Environment/archive/master.zip)
add_subdirectory(Arcade-Learning-Environment-master)

# Download ROMs
if(NOT IS_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/ROMS)
  message("Downloading ROMs")
  file(DOWNLOAD
      http://www.atarimania.com/roms/Roms.rar 
      ${CMAKE_SOURCE_DIR}/third_party/Roms.rar
      SHOW_PROGRESS)
  execute_process(COMMAND ${CMAKE_COMMAND} -E
    chdir ${CMAKE_SOURCE_DIR}/third_party
    unrar e ${CMAKE_SOURCE_DIR}/third_party/Roms.rar)
  execute_process(COMMAND ${CMAKE_COMMAND} -E
    chdir ${CMAKE_SOURCE_DIR}/third_party
    unzip e ${CMAKE_SOURCE_DIR}/third_party/ROMS.zip)
endif()