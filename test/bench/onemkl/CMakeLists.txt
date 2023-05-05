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

if(${SYCLFFT_ENABLE_INTEL_CLOSED_ONEMKL_BENCHMARKS})
    add_benchmark(bench_closed_onemkl bench_closed_onemkl.cpp)
    add_sycl_to_target(TARGET bench_closed_onemkl)
    find_package(MKL CONFIG REQUIRED)
    target_compile_options(bench_closed_onemkl PRIVATE ${MKL_COPT} -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(bench_closed_onemkl PRIVATE MKL::MKL ${MKL_LINK_SYCL})
endif()

if(SYCLFFT_INTEL_OPEN_ONEMKL_BENCHMARK_BACKEND)
    add_benchmark(bench_open_onemkl bench_open_onemkl.cpp)
    add_sycl_to_target(TARGET bench_open_onemkl)

    # Download and install open-source oneMKL
    # Only enable the DFT domain and one of its backend
    set(ENABLE_MKLCPU_BACKEND OFF)
    set(ENABLE_MKLGPU_BACKEND OFF)
    set(ENABLE_CUFFT_BACKEND OFF)

    if(${SYCLFFT_INTEL_OPEN_ONEMKL_BENCHMARK_BACKEND} STREQUAL "MKLCPU")
      set(ENABLE_MKLCPU_BACKEND ON)
    elseif(${SYCLFFT_INTEL_OPEN_ONEMKL_BENCHMARK_BACKEND} STREQUAL "MKLGPU")
      set(ENABLE_MKLGPU_BACKEND ON)
    elseif(${SYCLFFT_INTEL_OPEN_ONEMKL_BENCHMARK_BACKEND} STREQUAL "CUFFT")
      set(ENABLE_CUFFT_BACKEND ON)
    else()
      message(FATAL_ERROR "Unsupported backend: ${SYCLFFT_INTEL_OPEN_ONEMKL_BENCHMARK_BACKEND}")
    endif()

    include(ExternalProject)
    set(ONEMKL_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/3rd_party/src/oneMKLDownload-install)
    set(ONEMKL_INCLUDE_DIR ${ONEMKL_INSTALL_DIR}/include)
    set(ONEMKL_LIB_DIR ${ONEMKL_INSTALL_DIR}/lib)
    set(ONEMKL_LIB ${ONEMKL_LIB_DIR}/libonemkl.so)
    ExternalProject_Add(oneMKLDownload
        PREFIX 3rd_party
        GIT_REPOSITORY https://github.com/oneapi-src/oneMKL.git
        GIT_TAG a492e0acf01a9915f96659e84739e01754efe986
        CMAKE_ARGS
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
          -DCMAKE_GENERATOR=${CMAKE_GENERATOR}
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
          -DCMAKE_INSTALL_PREFIX=${ONEMKL_INSTALL_DIR}
          # Allow the library to dlopen the backend's library
          -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath=${ONEMKL_LIB_DIR}"
          -DTARGET_DOMAINS=dft
          -DENABLE_MKLCPU_BACKEND=${ENABLE_MKLCPU_BACKEND}
          -DENABLE_MKLGPU_BACKEND=${ENABLE_MKLGPU_BACKEND}
          -DENABLE_CUFFT_BACKEND=${ENABLE_CUFFT_BACKEND}
          -DBUILD_FUNCTIONAL_TESTS=OFF
          -DBUILD_EXAMPLES=OFF
          -DBUILD_DOC=OFF
        INSTALL_DIR ${ONEMKL_INSTALL_DIR}
        BUILD_BYPRODUCTS ${ONEMKL_LIB}
    )
    add_library(oneMKL SHARED IMPORTED)
    set_target_properties(oneMKL PROPERTIES IMPORTED_LOCATION ${ONEMKL_LIB})

    add_dependencies(bench_open_onemkl oneMKLDownload)
    target_link_libraries(bench_open_onemkl PRIVATE oneMKL)
    target_include_directories(bench_open_onemkl PRIVATE ${ONEMKL_INCLUDE_DIR})
endif()
