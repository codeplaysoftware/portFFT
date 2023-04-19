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

set(SYCLFFT_BENCHMARKS
    bench_float.cpp
    bench_manual_float.cpp
)
if(SYCLFFT_ENABLE_DOUBLE_BUILDS)
    list(APPEND SYCLFFT_BENCHMARKS
        bench_manual_double.cpp
    )
endif()

foreach(BENCHMARK_FILE ${SYCLFFT_BENCHMARKS})
    get_filename_component(FILE_NAME ${BENCHMARK_FILE} NAME_WE)
    set(BENCHMARK_TARGET "${FILE_NAME}")
    add_executable(
        ${BENCHMARK_TARGET}
        ${BENCHMARK_FILE}
    )
    add_sycl_to_target(TARGET ${BENCHMARK_TARGET} SOURCES ${BENCHMARK_FILE})
    if(${SYCLFFT_VERIFY_BENCHMARK})
        target_compile_definitions(${BENCHMARK_TARGET} PRIVATE SYCLFFT_VERIFY_BENCHMARK)
    endif()
    target_link_libraries(
        ${BENCHMARK_TARGET}
        PRIVATE
        sycl_fft
        benchmark::benchmark
        Threads::Threads
    )
    target_include_directories(${BENCHMARK_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/test/common)
endforeach()

if(${SYCLFFT_ENABLE_CUFFT_BENCHMARKS})
    find_package(CUDAToolkit REQUIRED)

    add_executable(bench_cufft bench_cufft.cpp)
    if(${SYCLFFT_VERIFY_BENCHMARK})
        target_compile_definitions(bench_cufft PRIVATE SYCLFFT_VERIFY_BENCHMARK)
    endif()
    target_link_libraries(bench_cufft PRIVATE CUDA::cudart CUDA::cufft benchmark::benchmark CUDA::curand sycl_fft)
    target_include_directories(bench_cufft PRIVATE ${PROJECT_SOURCE_DIR}/test/common)
endif()

if(${SYCLFFT_ENABLE_ROCFFT_BENCHMARKS})
    find_package(HIP REQUIRED)
    find_package(rocfft REQUIRED)

    # get target include directories from sycl-fft for the direction enum
    get_target_property(SYCLFFT_INCLUDES sycl_fft INTERFACE_INCLUDE_DIRECTORIES)

    add_executable(bench_rocfft bench_rocfft.cpp)
    target_link_libraries(bench_rocfft PRIVATE hip::host roc::rocfft benchmark::benchmark)
    target_include_directories(bench_rocfft PRIVATE ${SYCLFFT_INCLUDES} ${PROJECT_SOURCE_DIR}/test/common)

    if(${SYCLFFT_VERIFY_BENCHMARK})
        find_package(rocrand REQUIRED)

        target_compile_definitions(bench_rocfft PRIVATE SYCLFFT_VERIFY_BENCHMARK)
        target_link_libraries(bench_rocfft PRIVATE roc::rocrand)
    endif()
endif()

if(${SYCLFFT_ENABLE_INTEL_CLOSED_ONEMKL_BENCHMARKS})
    add_executable(bench_onemkl bench_onemkl.cpp)
    find_package(MKL CONFIG REQUIRED)
    target_compile_options(bench_onemkl PRIVATE ${MKL_COPT})
    target_link_libraries(bench_onemkl PRIVATE MKL::MKL ${MKL_LINK_SYCL} benchmark::benchmark)
    target_include_directories(bench_onemkl PRIVATE ${PROJECT_SOURCE_DIR}/test/common)
endif()
