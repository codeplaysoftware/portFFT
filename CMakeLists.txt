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
option(SYCLFFT_VERIFY_BENCHMARK "Whether to verify FFT results during benchmarking. Verifies in the first iteration only" OFF)
option(SYCLFFT_GENERATE_BENCH_REFERENCE_AT_BUILD_TIME "Generate benchmark verification data at build-time" ON)
option(SYCLFFT_ENABLE_DOUBLE_BUILDS "Enable building tests and benchmarks using double precision" ON)
option(SYCLFFT_USE_SG_TRANSFERS "Whether to use intel extension for subgroup joint loads and stores." OFF)
option(SYCLFFT_SLOW_SG_SHUFFLES "Whether subgroup shuffles are slow on target device and should be avoided." OFF)
set(SYCLFFT_TARGET_REGS_PER_WI 128 CACHE STRING "How many 32b registers can be allocated per work item on the target device")
set(SYCLFFT_SUBGROUP_SIZES 32 CACHE STRING "Comma separated list of subgroup sizes to compile for. The first size supported by the device will be used.")
set(SYCLFFT_TARGET_WI_LOAD 16 CACHE STRING "Number of consecutive bytes each work item should load at once.")
set(SYCLFFT_SGS_IN_WG 2 CACHE STRING "Number of subgroups per workgroup.")
set(SYCLFFT_DEVICE_TRIPLE "spir64" CACHE STRING "Specify the target triple representing target device architectures")
set(SYCLFFT_COOLEY_TUKEY_OPTIMIZED_SIZES "1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096,6144,8192" CACHE STRING "Specify the sizes supported in work-item, sub-group, and work-group Cooley-Tukey implementations")

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
target_compile_definitions(sycl_fft INTERFACE SYCLFFT_SUBGROUP_SIZES=${SYCLFFT_SUBGROUP_SIZES})
target_compile_definitions(sycl_fft INTERFACE SYCLFFT_TARGET_WI_LOAD=${SYCLFFT_TARGET_WI_LOAD})
target_compile_definitions(sycl_fft INTERFACE SYCLFFT_SGS_IN_WG=${SYCLFFT_SGS_IN_WG})
if(${SYCLFFT_USE_SG_TRANSFERS})
  target_compile_definitions(sycl_fft INTERFACE SYCLFFT_USE_SG_TRANSFERS)
endif()
if(${SYCLFFT_SLOW_SG_SHUFFLES})
  target_compile_definitions(sycl_fft INTERFACE SYCLFFT_SLOW_SG_SHUFFLES=1)
else()
  target_compile_definitions(sycl_fft INTERFACE SYCLFFT_SLOW_SG_SHUFFLES=0)
endif()

target_compile_options(sycl_fft INTERFACE -fgpu-inline-threshold=1000000)
target_link_options(sycl_fft INTERFACE -fsycl-device-code-split=per_kernel)
target_compile_options(sycl_fft INTERFACE -fsycl-device-code-split=per_kernel)

include(CMakePackageConfigHelpers)
set(version_file "${CMAKE_CURRENT_BINARY_DIR}/cmake/sycl_fft-version.cmake")
write_basic_package_version_file(${version_file}
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# Check that the following is a comma separated list.
if(SYCLFFT_COOLEY_TUKEY_OPTIMIZED_SIZES MATCHES "^[0-9]+(,[0-9]+)*$")
  message(STATUS "Building with Cooley-Tukey config: " ${SYCLFFT_COOLEY_TUKEY_OPTIMIZED_SIZES})
else()
  message(FATAL_ERROR " Invalid SYCLFFT_COOLEY_TUKEY_OPTIMIZED_SIZES value: " ${SYCLFFT_COOLEY_TUKEY_OPTIMIZED_SIZES})
endif()
target_compile_definitions(sycl_fft INTERFACE SYCLFFT_COOLEY_TUKEY_OPTIMIZED_SIZES=${SYCLFFT_COOLEY_TUKEY_OPTIMIZED_SIZES})

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

add_library(sycl_fft_warnings INTERFACE)
target_compile_options(sycl_fft_warnings INTERFACE -Wall -Wextra -Wshadow -Wconversion -Wpedantic)

if(${SYCLFFT_BUILD_TESTS})
  enable_testing()
endif()

add_subdirectory(test)
