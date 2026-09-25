include_guard(GLOBAL)

set(_tiktoken_c_root "${CMAKE_CURRENT_LIST_DIR}")

function(_tiktoken_c_add_imported target type library)
  if(EXISTS "${library}")
    add_library("${target}" "${type}" IMPORTED GLOBAL)
    set_target_properties("${target}" PROPERTIES
      IMPORTED_LOCATION "${library}"
      INTERFACE_INCLUDE_DIRECTORIES "${_tiktoken_c_root}"
    )
  endif()
endfunction()

if(WIN32)
  if(EXISTS "${_tiktoken_c_root}/tiktoken_c.dll")
    add_library(tiktoken-c::shared SHARED IMPORTED GLOBAL)
    set_target_properties(tiktoken-c::shared PROPERTIES
      IMPORTED_LOCATION "${_tiktoken_c_root}/tiktoken_c.dll"
      IMPORTED_IMPLIB "${_tiktoken_c_root}/tiktoken_c.dll.lib"
      INTERFACE_INCLUDE_DIRECTORIES "${_tiktoken_c_root}"
    )
  endif()
  _tiktoken_c_add_imported(tiktoken-c::static STATIC
    "${_tiktoken_c_root}/tiktoken_c.lib")

  if(EXISTS "${_tiktoken_c_root}/logging/tiktoken_c.dll")
    add_library(tiktoken-c::shared-logging SHARED IMPORTED GLOBAL)
    set_target_properties(tiktoken-c::shared-logging PROPERTIES
      IMPORTED_LOCATION "${_tiktoken_c_root}/logging/tiktoken_c.dll"
      IMPORTED_IMPLIB "${_tiktoken_c_root}/logging/tiktoken_c.dll.lib"
      INTERFACE_COMPILE_DEFINITIONS TIKTOKEN_C_ENABLE_LOGGING
      INTERFACE_INCLUDE_DIRECTORIES "${_tiktoken_c_root}"
    )
  endif()
  _tiktoken_c_add_imported(tiktoken-c::static-logging STATIC
    "${_tiktoken_c_root}/logging/tiktoken_c.lib")
elseif(APPLE)
  _tiktoken_c_add_imported(tiktoken-c::shared SHARED
    "${_tiktoken_c_root}/libtiktoken_c.dylib")
  _tiktoken_c_add_imported(tiktoken-c::static STATIC
    "${_tiktoken_c_root}/libtiktoken_c.a")
  _tiktoken_c_add_imported(tiktoken-c::shared-logging SHARED
    "${_tiktoken_c_root}/logging/libtiktoken_c.dylib")
  _tiktoken_c_add_imported(tiktoken-c::static-logging STATIC
    "${_tiktoken_c_root}/logging/libtiktoken_c.a")
else()
  _tiktoken_c_add_imported(tiktoken-c::shared SHARED
    "${_tiktoken_c_root}/libtiktoken_c.so")
  _tiktoken_c_add_imported(tiktoken-c::static STATIC
    "${_tiktoken_c_root}/libtiktoken_c.a")
  _tiktoken_c_add_imported(tiktoken-c::shared-logging SHARED
    "${_tiktoken_c_root}/logging/libtiktoken_c.so")
  _tiktoken_c_add_imported(tiktoken-c::static-logging STATIC
    "${_tiktoken_c_root}/logging/libtiktoken_c.a")

  foreach(_target tiktoken-c::shared tiktoken-c::shared-logging)
    if(TARGET "${_target}")
      set_property(TARGET "${_target}" PROPERTY IMPORTED_NO_SONAME TRUE)
    endif()
  endforeach()
endif()

foreach(_target tiktoken-c::shared-logging tiktoken-c::static-logging)
  if(TARGET "${_target}")
    set_property(TARGET "${_target}" APPEND PROPERTY
      INTERFACE_COMPILE_DEFINITIONS TIKTOKEN_C_ENABLE_LOGGING)
  endif()
endforeach()

if(WIN32)
  foreach(_target tiktoken-c::static tiktoken-c::static-logging)
    if(TARGET "${_target}")
      set_property(TARGET "${_target}" APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES ntdll)
    endif()
  endforeach()
endif()

if(UNIX AND TARGET tiktoken-c::static)
  find_package(Threads REQUIRED)
  foreach(_target tiktoken-c::static tiktoken-c::static-logging)
    if(TARGET "${_target}")
      set_property(TARGET "${_target}" APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS};m")
      if(APPLE)
        set_property(TARGET "${_target}" APPEND PROPERTY
          INTERFACE_LINK_LIBRARIES iconv)
      else()
        set_property(TARGET "${_target}" APPEND PROPERTY
          INTERFACE_LINK_LIBRARIES "rt;util;gcc_s")
      endif()
    endif()
  endforeach()
endif()

if(TARGET tiktoken-c::shared)
  add_library(tiktoken-c::tiktoken-c INTERFACE IMPORTED GLOBAL)
  set_property(TARGET tiktoken-c::tiktoken-c PROPERTY
    INTERFACE_LINK_LIBRARIES tiktoken-c::shared)
elseif(TARGET tiktoken-c::static)
  add_library(tiktoken-c::tiktoken-c INTERFACE IMPORTED GLOBAL)
  set_property(TARGET tiktoken-c::tiktoken-c PROPERTY
    INTERFACE_LINK_LIBRARIES tiktoken-c::static)
else()
  set(tiktoken-c_FOUND FALSE)
  set(tiktoken-c_NOT_FOUND_MESSAGE
    "No tiktoken-c library was found in ${_tiktoken_c_root}")
endif()

unset(_tiktoken_c_root)
