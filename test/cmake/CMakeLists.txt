cmake_minimum_required(VERSION 3.15)
project(tiktoken_c_package_test LANGUAGES C)

find_package(tiktoken-c CONFIG REQUIRED)

add_executable(tiktoken_c_package_test main.c)
target_link_libraries(tiktoken_c_package_test PRIVATE tiktoken-c::tiktoken-c)

add_executable(tiktoken_c_static_package_test main.c)
target_link_libraries(tiktoken_c_static_package_test PRIVATE tiktoken-c::static)

add_executable(tiktoken_c_logging_package_test main.c)
target_link_libraries(tiktoken_c_logging_package_test PRIVATE tiktoken-c::shared-logging)

add_executable(tiktoken_c_logging_static_package_test main.c)
target_link_libraries(tiktoken_c_logging_static_package_test PRIVATE tiktoken-c::static-logging)

enable_testing()
add_test(NAME shared_package COMMAND tiktoken_c_package_test)
add_test(NAME static_package COMMAND tiktoken_c_static_package_test)
add_test(NAME shared_logging_package COMMAND tiktoken_c_logging_package_test)
add_test(NAME static_logging_package COMMAND tiktoken_c_logging_static_package_test)

if(WIN32)
  get_target_property(_shared_library tiktoken-c::shared IMPORTED_LOCATION)
  get_filename_component(_shared_directory "${_shared_library}" DIRECTORY)
  get_target_property(_logging_library tiktoken-c::shared-logging IMPORTED_LOCATION)
  get_filename_component(_logging_directory "${_logging_library}" DIRECTORY)
  string(REPLACE ";" "\\;" _system_path "$ENV{PATH}")
  set_tests_properties(shared_package PROPERTIES
    ENVIRONMENT "PATH=${_shared_directory}\\;${_system_path}")
  set_tests_properties(shared_logging_package PROPERTIES
    ENVIRONMENT "PATH=${_logging_directory}\\;${_system_path}")
endif()
