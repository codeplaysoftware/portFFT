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

set(PORTFFT_BENCHMARKS
    bench_float.cpp
    bench_manual_float.cpp
)
if(PORTFFT_ENABLE_DOUBLE_BUILDS)
    list(APPEND PORTFFT_BENCHMARKS
        bench_manual_double.cpp
    )
endif()

foreach(BENCHMARK_FILE ${PORTFFT_BENCHMARKS})
    get_filename_component(BENCHMARK_TARGET ${BENCHMARK_FILE} NAME_WE)
    add_benchmark(${BENCHMARK_TARGET} ${BENCHMARK_FILE})
    add_sycl_to_target(TARGET ${BENCHMARK_TARGET})
    target_link_libraries(${BENCHMARK_TARGET} PRIVATE portfft)
endforeach()
