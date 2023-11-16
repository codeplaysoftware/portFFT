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

cmake_minimum_required(VERSION 3.20...3.26)
project(portFFT VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
)

option(PORTFFT_BUILD_TESTS "Whether to enable building tests" OFF)
option(PORTFFT_BUILD_BENCHMARKS "Whether to enable building benchmarks" OFF)
option(PORTFFT_VERIFY_BENCHMARKS "Whether to verify FFT results during benchmarking. Verifies in the first iteration only" OFF)
option(PORTFFT_ENABLE_DOUBLE_BUILDS "Enable building tests and benchmarks using double precision" ON)
option(PORTFFT_USE_SG_TRANSFERS "Whether to use intel extension for subgroup joint loads and stores." OFF)
option(PORTFFT_SLOW_SG_SHUFFLES "Whether subgroup shuffles are slow on target device and should be avoided." OFF)
option(PORTFFT_USE_SCLA "Whether to use spec-constant length array (experimental)" OFF)
option(PORTFFT_CLANG_TIDY "Enable clang-tidy checks on portFFT source when building tests" ON)
option(PORTFFT_CLANG_TIDY_AUTOFIX "Attempt to fix defects found by clang-tidy" OFF)
option(PORTFFT_LOG_DUMPS "Whether to enable logging of data dumps" OFF)
option(PORTFFT_LOG_TRANSFERS "Whether to enable logging of memory transfers" OFF)
option(PORTFFT_LOG_TRACE "Whether to enable tracing of function calls" OFF)
set(PORTFFT_REGISTERS_PER_WI 128 CACHE STRING "How many 32b registers can be allocated per work item on the target device")
set(PORTFFT_SUBGROUP_SIZES 32 CACHE STRING "Comma separated list of subgroup sizes to compile for. The first size supported by the device will be used.")
set(PORTFFT_VEC_LOAD_BYTES 16 CACHE STRING "Number of consecutive bytes each work item should load at once.")
set(PORTFFT_SGS_IN_WG 2 CACHE STRING "Number of subgroups per workgroup.")
set(PORTFFT_MAX_CONCURRENT_KERNELS 16 CACHE STRING "Maximum number of resident kernels possible on the hardware")
set(PORTFFT_DEVICE_TRIPLE "spir64" CACHE STRING "Specify the target triple representing target device architectures")
set(PORTFFT_CLANG_OPTIMIZATION_REMARKS_REGEX "" CACHE STRING "Use -fsave-optimization-record -Rpass-missed=<regex> -Rpass=<regex> -Rpass-analysis=<regex> to obtain optimization pass remarks. See https://llvm.org/docs/Passes.html for passes.")

set(PORTFFT_INCLUDE_DIR
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
set(PORTFFT_SRC_DIR
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:src>)

add_library(portfft INTERFACE)
target_include_directories(portfft INTERFACE
    ${PORTFFT_INCLUDE_DIR}
    ${PORTFFT_SRC_DIR}
)
target_compile_definitions(portfft INTERFACE PORTFFT_REGISTERS_PER_WI=${PORTFFT_REGISTERS_PER_WI})
target_compile_definitions(portfft INTERFACE PORTFFT_SUBGROUP_SIZES=${PORTFFT_SUBGROUP_SIZES})
target_compile_definitions(portfft INTERFACE PORTFFT_VEC_LOAD_BYTES=${PORTFFT_VEC_LOAD_BYTES})
target_compile_definitions(portfft INTERFACE PORTFFT_SGS_IN_WG=${PORTFFT_SGS_IN_WG})
if(${PORTFFT_USE_SG_TRANSFERS})
  target_compile_definitions(portfft INTERFACE PORTFFT_USE_SG_TRANSFERS)
endif()
if(${PORTFFT_SLOW_SG_SHUFFLES})
  target_compile_definitions(portfft INTERFACE PORTFFT_SLOW_SG_SHUFFLES=1)
else()
  target_compile_definitions(portfft INTERFACE PORTFFT_SLOW_SG_SHUFFLES=0)
endif()
if(${PORTFFT_USE_SCLA})
  target_compile_definitions(portfft INTERFACE PORTFFT_USE_SCLA)
endif()

set(PORTFFT_LOG OFF)
if(${PORTFFT_LOG_DUMPS})
  target_compile_definitions(portfft INTERFACE PORTFFT_LOG_DUMPS)
  set(PORTFFT_LOG ON)
endif()
if(${PORTFFT_LOG_TRANSFERS})
  target_compile_definitions(portfft INTERFACE PORTFFT_LOG_TRANSFERS)
  set(PORTFFT_LOG ON)
endif()
if(${PORTFFT_LOG_TRACE})
  target_compile_definitions(portfft INTERFACE PORTFFT_LOG_TRACE)
  set(PORTFFT_LOG ON)
endif()
if(${PORTFFT_LOG})
  target_compile_definitions(portfft INTERFACE PORTFFT_LOG)
endif()
if(${PORTFFT_MAX_CONCURRENT_KERNELS})
  target_compile_definitions(portfft INTERFACE PORTFFT_MAX_CONCURRENT_KERNELS)
endif()

target_compile_options(portfft INTERFACE -fgpu-inline-threshold=1000000)
target_link_options(portfft INTERFACE -fsycl-device-code-split=per_kernel)
target_compile_options(portfft INTERFACE -fsycl-device-code-split=per_kernel)

include(CMakePackageConfigHelpers)
set(version_file "${CMAKE_CURRENT_BINARY_DIR}/cmake/portfft-version.cmake")
write_basic_package_version_file(${version_file}
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

include(GNUInstallDirs)
install(TARGETS portfft
  EXPORT portfft
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
  DIRECTORY
    ${PORTFFT_INCLUDE_DIR}
    ${PORTFFT_SRC_DIR}
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  COMPONENT portfft
  FILES_MATCHING PATTERN "*.hpp"
)

install(FILES ${version_file} DESTINATION ${CMAKE_INSTALL_PREFIX})
install(EXPORT portfft
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  NAMESPACE portfft::
  FILE portfft-config.cmake
)

export(EXPORT portfft
  NAMESPACE portfft::
  FILE portfft-config.cmake
)

add_library(portfft_warnings INTERFACE)
target_compile_options(portfft_warnings INTERFACE -Wall -Wextra -Wshadow -Wconversion -Wpedantic -Wno-intel-compat)

if(${PORTFFT_BUILD_TESTS})
  enable_testing()
endif()

add_subdirectory(test)
