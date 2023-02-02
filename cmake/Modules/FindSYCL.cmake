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
# *  Codeplay's SYCL-FFT
# *
# *  @filename CMakeLists.txt
# *
# **************************************************************************/


include_guard()

# Try to find a DPC++ release 
# (reqrs source /opt/intel/oneapi/compilers/2023.0.0/env/vars.sh)
find_package(IntelDPCPP QUIET)
if(IntelDPCPP_FOUND)
    function(add_sycl_to_target)            
        # SYCL is already added to targets for DPC++ release.     
    endfunction()
endif()

# Try to find DPC++ (nightly or manually set compiler path)
if(NOT IntelDPCPP_FOUND)
    find_package(DPCPP QUIET)
endif()

# If DPC++ hasn't already been set at the command line, try finding ComputeCpp:
if(NOT IntelDPCPP_FOUND AND NOT DPCPP_FOUND)
    set(SYCL_LANGUAGE_VERSION 2020)
    set(COMPUTECPP_BITCODE spirv64)
    find_package(ComputeCpp QUIET)
endif()

if(NOT ComputeCpp_FOUND AND NOT IntelDPCPP_FOUND AND NOT DPCPP_FOUND)
  # Display warnings
  find_package(ComputeCpp)
  find_package(DPCPP)
  message(FATAL_ERROR "No SYCL implementation found")
endif()
