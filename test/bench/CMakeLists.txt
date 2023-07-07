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

if(${SYCLFFT_VERIFY_BENCHMARK})
    # Generate benchmark reference data header
    execute_process(
        COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/generate_reference_data.py --build_path=${CMAKE_BINARY_DIR} --header --benchmark --just_list_outputs 
        OUTPUT_VARIABLE BENCHMARK_REFERENCE_DATA_HEADER)
    add_custom_command(
        OUTPUT ${BENCHMARK_REFERENCE_DATA_HEADER}
        COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/generate_reference_data.py --build_path=${CMAKE_BINARY_DIR} --header --benchmark)
    add_custom_target(benchmark_reference_data_header 
        DEPENDS ${BENCHMARK_REFERENCE_DATA_HEADER})
    # We need the directory so the header can be included by the benchmark.
    get_filename_component(BENCHMARK_REFERENCE_DATA_HEADER_DIR ${BENCHMARK_REFERENCE_DATA_HEADER} DIRECTORY)

    # Generate the actual test reference data
    execute_process(
        COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/generate_reference_data.py --build_path=${CMAKE_BINARY_DIR} --data --benchmark --just_list_outputs 
        OUTPUT_VARIABLE BENCHMARK_REFERENCE_DATA_DATA_FILES)
    add_custom_command(
        OUTPUT ${BENCHMARK_REFERENCE_DATA_DATA_FILES}
        COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/generate_reference_data.py --build_path=${CMAKE_BINARY_DIR} --data --benchmark)
    add_custom_target(benchmark_reference_data_data 
        DEPENDS ${BENCHMARK_REFERENCE_DATA_DATA_FILES})
endif()

# Common function to add a benchmark
set(BENCHMARK_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}")
function(add_benchmark target source_file)
  add_executable(${target} ${source_file})
  target_link_libraries(${target} PRIVATE
      benchmark::benchmark
      Threads::Threads
      sycl_fft_warnings
  )

  # get target include directories from sycl-fft for the direction enum
  get_target_property(SYCLFFT_INCLUDES sycl_fft INTERFACE_INCLUDE_DIRECTORIES)

  target_include_directories(${target} PRIVATE
      ${PROJECT_SOURCE_DIR}/test/common
      ${PROJECT_SOURCE_DIR}/test/bench/utils
      ${SYCLFFT_INCLUDES}
  )
  if(${SYCLFFT_VERIFY_BENCHMARK})
      target_compile_definitions(${target} PRIVATE SYCLFFT_VERIFY_BENCHMARK)
      target_include_directories(${target} PRIVATE ${BENCHMARK_REFERENCE_DATA_HEADER_DIR})
      add_dependencies(${target} benchmark_reference_data_header benchmark_reference_data_data)
  endif()
  set_target_properties(${target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${BENCHMARK_BIN_DIR}
  )
endfunction()

add_subdirectory(sycl-fft)
