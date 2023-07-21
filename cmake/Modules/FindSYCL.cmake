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
# (reqrs source /opt/intel/oneapi/compilers/2023.1.0/env/vars.sh)
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
    target_compile_options(${ARG_TARGET} PUBLIC ${COMPILE_FLAGS})
    target_link_options(${ARG_TARGET} PUBLIC ${COMPILE_FLAGS})
    endfunction()
endif()

# Try to find DPC++ (nightly or manually set compiler path)
if(NOT IntelSYCL_FOUND)
    find_package(DPCPP QUIET)
endif()

if(NOT IntelSYCL_FOUND AND NOT DPCPP_FOUND)
  # Display warnings
  find_package(IntelSYCL)
  find_package(DPCPP)
  message(FATAL_ERROR "No SYCL implementation found")
endif()
