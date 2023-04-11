/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  A set of common reference DFT benchmarks.
 *
 **************************************************************************/
#ifndef SYCL_FFT_REFERENCE_DFT_SET_HPP
#define SYCL_FFT_REFERENCE_DFT_SET_HPP

/**
 * A common set of reference benchmarks. To use, two macros must be defined:
 * BENCH_COMPLEX_FLOAT(...)
 * BENCH_SINGLE_FLOAT(...)
 * and then the benchmarks can be instantiated:
 * INSTANTIATE_REFERENCE_BENCHMARK_SET(BENCH_COMPLEX_FLOAT, BENCH_SINGLE_FLOAT)
 * See pre-existing benchmark implementations for examples of what these
 * macros should do.
 **/

#include <vector>

// clang-format off
// Forward, float, out-of-place only:
// 1. small        complex 1D fits in workitem Cooley-Tukey 	   (batch=8*1024*1024 N=16)
// 2. medium-small complex 1D fits in subgroup Cooley-Tukey 	   (batch=512*1024    N=256)
// 3. medium-large complex 1D fits in local memory Cooley-Tukey    (batch=32*1024     N=4*1024)
// 4. large        complex 1D fits in global memory Cooley-Tukey   (batch=2*1024      N=64*1024)
// 5. large        complex 1D fits in global memory Bluestein      (batch=2*1024      N=64*1024+1)
// 6. large        complex 2D fits in global memory                (batch=8           N=4096x4096)
// 7. small        real    1D fits in workitem Cooley-Tukey 	   (batch=8*1024*1024 N=32)
// 8. medium-small real    1D fits in subgroup Cooley-Tukey 	   (batch=512*1024    N=512)
// 9. medium-large real    1D fits in local memory Cooley-Tukey    (batch=32*1024     N=8*1024)
// 10. large       real    1D fits in global memory Cooley-Tukey   (batch=2*1024      N=128*1024)
// 11. small       real    3D                                      (batch=1024        N=64x64x64)

// Arguments: N, batch
#define INSTANTIATE_REFERENCE_BENCHMARK_SET(bench_complex_float_macro, bench_single_float_macro)    \
bench_complex_float_macro(small_1d,        std::vector<int>({16}),            8 * 1024 * 1024);     \
bench_complex_float_macro(medium_small_1d, std::vector<int>({256}),           512 * 1024);          \
bench_complex_float_macro(medium_large_1d, std::vector<int>({4 * 1024}),      32 * 1024);           \
bench_complex_float_macro(large_1d,        std::vector<int>({64 * 1024}),     2 * 1024);            \
bench_complex_float_macro(large_1d_prime,  std::vector<int>({64 * 1024 + 1}), 2 * 1024);            \
bench_complex_float_macro(large_2d,        std::vector<int>({4096, 4096}),    8);                   \
                                                                                                    \
bench_single_float_macro(small_1d,        std::vector<int>({32}),         8 * 1024 * 1024);         \
bench_single_float_macro(medium_small_1d, std::vector<int>({512}),        512 * 1024);              \
bench_single_float_macro(medium_large_1d, std::vector<int>({8 * 1024}),   32 * 1024);               \
bench_single_float_macro(large_1d,        std::vector<int>({128 * 1024}), 2 * 1024);                \
bench_single_float_macro(small_3d,        std::vector<int>({64, 64, 64}), 1024);
// clang-format on

#endif  // SYCL_FFT_REFERENCE_DFT_SET_HPP
