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

# Try to find a DPC++ release
# (reqrs source /path/to/intel/oneapi/compiler/2024.0/env/vars.sh)
find_package(IntelSYCL QUIET)
if(IntelSYCL_FOUND)
    function(add_sycl_to_target)
    set(options)
    set(one_value_args TARGET)
    cmake_parse_arguments(ARG
      "${options}"
      "${one_value_args}"
      "${multi_value_args}"
      ${ARGN}
    )
    set(COMPILE_FLAGS "-fsycl;-fsycl-targets=${PORTFFT_DEVICE_TRIPLE};-fsycl-unnamed-lambda")
    if(NOT PORTFFT_CLANG_OPTIMIZATION_REMARKS_REGEX STREQUAL "")
      STRING(REPLACE "<regex>" "${PORTFFT_CLANG_OPTIMIZATION_REMARKS_REGEX}" REMARK_ADDITIONAL_FLAGS "-fsave-optimization-record;-Rpass-missed=<regex>;-Rpass=<regex>;-Rpass-analysis=<regex>;")
      LIST(PREPEND COMPILE_FLAGS ${REMARK_ADDITIONAL_FLAGS})
    endif()
    target_compile_options(${ARG_TARGET} PUBLIC ${COMPILE_FLAGS})
    target_link_options(${ARG_TARGET} PUBLIC ${COMPILE_FLAGS})
    endfunction()
    return()
endif()

# Try to find DPC++ (nightly or manually set compiler path)
find_package(DPCPP QUIET)
if(DPCPP_FOUND)
  return()
endif()

# Try to find AdaptiveCpp
find_package(AdaptiveCpp QUIET)
if(AdaptiveCpp_FOUND)
  return()
endif()

# Display warnings
find_package(IntelSYCL)
find_package(DPCPP)
find_package(AdaptiveCpp)
message(FATAL_ERROR "No SYCL implementation found")
