if(NOT DEFINED TIKTOKEN_C_VERSION OR NOT DEFINED OUTPUT_DIR OR
   NOT DEFINED TIKTOKEN_C_STATIC_LIBS)
  message(FATAL_ERROR
    "TIKTOKEN_C_VERSION, TIKTOKEN_C_STATIC_LIBS, and OUTPUT_DIR are required")
endif()

include(CMakePackageConfigHelpers)

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/tiktoken-c-config.cmake"
  "${OUTPUT_DIR}/tiktoken-c-config.cmake"
  COPYONLY
)
write_basic_package_version_file(
  "${OUTPUT_DIR}/tiktoken-c-config-version.cmake"
  VERSION "${TIKTOKEN_C_VERSION}"
  COMPATIBILITY SameMinorVersion
)
configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/tiktoken-c.pc.in"
  "${OUTPUT_DIR}/tiktoken-c.pc"
  @ONLY
)
configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/tiktoken-c-logging.pc.in"
  "${OUTPUT_DIR}/tiktoken-c-logging.pc"
  @ONLY
)
configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/tiktoken-c-static.pc.in"
  "${OUTPUT_DIR}/tiktoken-c-static.pc"
  @ONLY
)
configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/tiktoken-c-logging-static.pc.in"
  "${OUTPUT_DIR}/tiktoken-c-logging-static.pc"
  @ONLY
)
