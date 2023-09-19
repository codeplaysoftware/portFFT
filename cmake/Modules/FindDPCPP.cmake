#/***************************************************************************
# *
# *  @license
# *  Copyright (C) Codeplay Software Limited
# *  Licensed under the Apache License, Version 2.0 (the "License");
# *  you may not use this file except in compliance with the License.
# *  You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# *  For your convenience, a copy of the License has been included in this
# *  repository.
# *
# *  Unless required by applicable law or agreed to in writing, software
# *  distributed under the License is distributed on an "AS IS" BASIS,
# *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *  See the License for the specific language governing permissions and
# *  limitations under the License.
# *
# *  Codeplay's portFFT
# *
# *  @filename CMakeLists.txt
# *
# **************************************************************************/

include_guard()

include(FindPackageHandleStandardArgs)

get_filename_component(DPCPP_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
find_library(DPCPP_LIB NAMES sycl sycl6 PATHS "${DPCPP_BIN_DIR}/../lib")

find_package_handle_standard_args(DPCPP
  FOUND_VAR     DPCPP_FOUND
  REQUIRED_VARS DPCPP_LIB
)

if(NOT DPCPP_FOUND)
  return()
endif()

mark_as_advanced(DPCPP_FOUND DPCPP_LIB)

if(DPCPP_FOUND AND NOT TARGET DPCPP::DPCPP)
  set(CMAKE_CXX_STANDARD 17)
  add_library(DPCPP::DPCPP INTERFACE IMPORTED)
  set(COMPILE_FLAGS "-fsycl;-fsycl-targets=${PORTFFT_DEVICE_TRIPLE};-fsycl-unnamed-lambda")
  set_target_properties(DPCPP::DPCPP PROPERTIES
    INTERFACE_COMPILE_OPTIONS ${COMPILE_FLAGS}
    INTERFACE_LINK_OPTIONS ${COMPILE_FLAGS}
    INTERFACE_LINK_LIBRARIES ${DPCPP_LIB}
    INTERFACE_INCLUDE_DIRECTORIES "${DPCPP_BIN_DIR}/../include/sycl;${DPCPP_BIN_DIR}/../include")
endif()

function(add_sycl_to_target)
  set(options)
  set(one_value_args TARGET)
  cmake_parse_arguments(ARG
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(ADDITIONAL_FLAGS "")
  if(NOT PORTFFT_CLANG_OPTIMIZATION_REMARKS_REGEX STREQUAL "")
    STRING(REPLACE "<regex>" "${PORTFFT_CLANG_OPTIMIZATION_REMARKS_REGEX}" REMARK_ADDITIONAL_FLAGS "-fsave-optimization-record;-Rpass-missed=<regex>;-Rpass=<regex>;-Rpass-analysis=<regex>;")
    LIST(PREPEND ADDITIONAL_FLAGS ${REMARK_ADDITIONAL_FLAGS})
    message(STATUS Flags... ${ADDITIONAL_FLAGS})
  endif()
  target_link_libraries(${ARG_TARGET} PUBLIC DPCPP::DPCPP)
  target_compile_options(${ARG_TARGET} PUBLIC ${ADDITIONAL_FLAGS})
  target_link_options(${ARG_TARGET} PUBLIC ${ADDITIONAL_FLAGS})
endfunction()
