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

find_package(SYCL)

# Disable build of unit-tests for googlebenchmark:
set(BENCHMARK_ENABLE_TESTING OFF)
# The icpx compiler causes errors when in release mode:
set(BENCHMARK_ENABLE_WERROR OFF)
# Fetch googlebenchmark:
include(FetchContent)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.7.1
)
FetchContent_MakeAvailable(googlebenchmark)

# Common function to add a benchmark
set(BENCHMARK_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}")
function(add_benchmark target source_file)
  add_executable(${target} ${source_file})
  target_link_libraries(${target} PRIVATE
      benchmark::benchmark
      Threads::Threads
      portfft_warnings
  )

  # get target include directories from portfft for the direction enum
  get_target_property(PORTFFT_INCLUDES portfft INTERFACE_INCLUDE_DIRECTORIES)

  target_include_directories(${target} PRIVATE
      ${PROJECT_SOURCE_DIR}/test/common
      ${PROJECT_SOURCE_DIR}/test/bench/utils
      ${PORTFFT_INCLUDES}
  )

  if(PORTFFT_VERIFY_BENCHMARKS)
      target_compile_definitions(${target} PRIVATE PORTFFT_VERIFY_BENCHMARKS)
  endif()

  set_target_properties(${target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${BENCHMARK_BIN_DIR}
  )
endfunction()

add_subdirectory(portfft)
