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

cmake_minimum_required (VERSION 3.16)
project (sycl-fft VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
)

option(SYCLFFT_BUILD_TESTS "Whether to enable building tests" OFF)
option(SYCLFFT_BUILD_BENCHMARKS "Whether to enable building benchmarks" OFF)

add_library(sycl-fft INTERFACE)
target_include_directories(sycl-fft 
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(${SYCLFFT_BUILD_TESTS})
  enable_testing()
  add_subdirectory(test)
endif()

if(${SYCLFFT_BUILD_BENCHMARKS})
  add_subdirectory(bench)
endif()
