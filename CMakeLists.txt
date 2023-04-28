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

cmake_minimum_required(VERSION 3.16...3.26)
project(sycl-fft VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
)

option(SYCLFFT_BUILD_TESTS "Whether to enable building tests" OFF)
option(SYCLFFT_BUILD_BENCHMARKS "Whether to enable building benchmarks" OFF)
option(SYCLFFT_VERIFY_BENCHMARK "Whether to verify FFT results during benchmarking. Verifies in the first iteration only" ON)
option(SYCLFFT_ENABLE_DOUBLE_BUILDS "Enable building tests and benchmarks using double precision" ON)
option(SYCLFFT_ENABLE_CUFFT_BENCHMARKS "Whether to also build the cuFFT benchmark for comparison" OFF)
option(SYCLFFT_ENABLE_ROCFFT_BENCHMARKS "Whether to also build the rocFFT benchmark for comparison" OFF)
option(SYCLFFT_ENABLE_INTEL_CLOSED_ONEMKL_BENCHMARKS "Whether to also build the closed-source Intel OneMKL benchmark for comparison" OFF)
set(SYCLFFT_INTEL_OPEN_ONEMKL_BENCHMARK_BACKEND CACHE STRING "Build the open-source Intel OneMKL benchmark for comparison. Must be set to one of `MKLCPU`, `MKLGPU` or `CUFFT`.")
set(SYCLFFT_TARGET_REGS_PER_WI 128 CACHE STRING "How many 32b registers can be allocated per work item on the target device")
set(SYCLFFT_TARGET_SUBGROUP_SIZE 32 CACHE STRING "Size of the subgroup on the target device")
set(SYCLFFT_TARGET_WG_LOAD 16 CACHE STRING "Number of consecutive bytes each work item should load at once.")
set(SYCLFFT_DEVICE_TRIPLE "spir64" CACHE STRING "Specify the target triple representing target device architectures")

set(SYCLFFT_INCLUDE_DIR
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
set(SYCLFFT_SRC_DIR
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:src>)

add_library(sycl_fft INTERFACE)
target_include_directories(sycl_fft INTERFACE
    ${SYCLFFT_INCLUDE_DIR}
    ${SYCLFFT_SRC_DIR}
)
target_compile_definitions(sycl_fft INTERFACE SYCLFFT_TARGET_REGS_PER_WI=${SYCLFFT_TARGET_REGS_PER_WI})
target_compile_definitions(sycl_fft INTERFACE SYCLFFT_TARGET_SUBGROUP_SIZE=${SYCLFFT_TARGET_SUBGROUP_SIZE})
target_compile_definitions(sycl_fft INTERFACE SYCLFFT_TARGET_WG_LOAD=${SYCLFFT_TARGET_WG_LOAD})

include(CMakePackageConfigHelpers)
set(version_file "${CMAKE_CURRENT_BINARY_DIR}/cmake/sycl_fft-version.cmake")
write_basic_package_version_file(${version_file}
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

include(GNUInstallDirs)
install(TARGETS sycl_fft
  EXPORT sycl_fft
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  DIRECTORY
    ${SYCLFFT_INCLUDE_DIR}
    ${SYCLFFT_SRC_DIR}
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  COMPONENT sycl_fft
  FILES_MATCHING PATTERN "*.hpp"
)

install(FILES ${version_file} DESTINATION ${CMAKE_INSTALL_PREFIX})
install(EXPORT sycl_fft
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  NAMESPACE sycl_fft::
  FILE sycl_fft-config.cmake
)

export(EXPORT sycl_fft
  NAMESPACE sycl_fft::
  FILE sycl_fft-config.cmake
)

if(${SYCLFFT_BUILD_TESTS})
  enable_testing()
endif()

add_subdirectory(test)
