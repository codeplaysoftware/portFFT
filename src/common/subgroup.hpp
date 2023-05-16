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
 *  Codeplay's SYCL-FFT
 *
 **************************************************************************/

#ifndef SYCL_FFT_COMMON_SUBGROUP_HPP
#define SYCL_FFT_COMMON_SUBGROUP_HPP

#include <common/helpers.hpp>
#include <common/twiddle.hpp>
#include <common/twiddle_calc.hpp>
#include <common/workitem.hpp>
#include <enums.hpp>
#include <sycl/sycl.hpp>

namespace sycl_fft {
namespace detail {

/*
`sg_dft` calculates a DFT by a subgroup on values that are already loaded into private memory of the workitems in the
subgroup. It needs twiddle factors precalculated by `sg_calc_twiddles`. It handles the first factor by cross subgroup
DFT calling `cross_sg_dispatcher` and the second one by workitem implementation - calling `wi_dft`. It does twiddle
multiplication inbetween, but does not transpose. Transposition is supposed to be done when storing the values back to
the local memory.

The size of the DFT performed by this function is `N * M` - for the arguments `N` and `M`. `N` workitems work jointly on
one DFT, so at most `subgroup_size / N` DFTs can be performed by one subgroup at a time. If `N` does not evenly divide
`subgroup_size`, extra workitems perform dummy computations. However, they must also call `sg_dft`, as it uses group
functions.

On input, each of the `N` workitems hold `M` consecutive complex input values. On output, each of the workitems holds
complex values that are strided with stride `N` and consecutive workitems have consecutive values.

`cross_sg_dispatcher` selects the appropriate size for calling `cross_sg_dft` - making that size compile time constant.

`cross_sg_dft` calculates DFT across workitems, with each workitem contributing one complex value as input and output of
the computation. If the size of the subgroup is large enough compared to FFT size, a subgroup can calculate multiple
DFTs at once (the same holds true for `cross_sg_cooley_tukey_dft` and `cross_sg_naive_dft`). It calls either
`cross_sg_cooley_tukey_dft` (for composite sizes) or `cross_sg_naive_dft` (for prime sizes).

`cross_sg_cooley_tukey_dft` calculates DFT of a composite size across workitems. It calls `cross_sg_dft` for each of the
factors and does transposition and twiddle multiplication inbetween.

`cross_sg_naive_dft` calculates DFT across workitems using naive DFT algorithm.
*/

// forward declaration
template <direction dir, int N, int stride, typename T>
inline void cross_sg_dft(T& real, T& imag, sycl::sub_group& sg);

/**
 * Calculates DFT using naive algorithm by using workitems of one subgroup.
 * Each workitem holds one input and one output complex value.
 *
 * @tparam dir direction of the FFT
 * @tparam N size of the DFT transform
 * @tparam stride stride between workitems working on consecutive values of one
 * DFT
 * @tparam T type of the scalar to work on
 * @param[in,out] real real component of the input/output complex value for one
 * workitem
 * @param[in,out] imag imaginary component of the input/output complex value for
 * one workitem
 * @param sg subgroup
 */
template <direction dir, int N, int stride, typename T>
__attribute__((always_inline))  __attribute__((flatten)) inline void cross_sg_naive_dft(T& real, T& imag, sycl::sub_group& sg) {
  int local_id = sg.get_local_linear_id();
  int idx_out = (local_id / stride) % N;
  int fft_start = local_id - idx_out * stride;

  T res_real = 0;
  T res_imag = 0;

  unrolled_loop<0, N, 1>([&](int idx_in) __attribute__((always_inline))  __attribute__((flatten)) {
    const T multi_re = twiddle<T>::re[N][idx_in * idx_out % N];
    const T multi_im = [&]() __attribute__((always_inline))  __attribute__((flatten)) {
      if constexpr (dir == direction::FORWARD) return twiddle<T>::im[N][idx_in * idx_out % N];
      return -twiddle<T>::im[N][idx_in * idx_out % N];
    }();
    int source_wi_id = fft_start + idx_in * stride;

    T cur_real = sycl::select_from_group(sg, real, source_wi_id);
    T cur_imag = sycl::select_from_group(sg, imag, source_wi_id);

    // multiply cur and multi
    res_real += cur_real * multi_re - cur_imag * multi_im;
    res_imag += cur_real * multi_im + cur_imag * multi_re;
  });

  real = res_real;
  imag = res_imag;
}

/**
 * Transposes values held by workitems of a subgroup. Transposes rectangles of
 * size N*M. Each of the rectangles can be strided.
 *
 * @tparam N inner - contiguous size on input, outer size on output
 * @tparam M outer size on input, inner - contiguous size on output
 * @tparam stride stride between consecutive values of one rectangle
 * @tparam T type of the scalar to work on
 * @param[in,out] real real component of the input/output complex value for one
 * workitem
 * @param[in,out] imag imaginary component of the input/output complex value for
 * one workitem
 * @param sg subgroup
 */
template <int N, int M, int stride, typename T>
__attribute__((always_inline))  __attribute__((flatten)) inline void cross_sg_transpose(T& real, T& imag, sycl::sub_group& sg) {
  int local_id = sg.get_local_linear_id();
  int index_in_outer_dft = (local_id / stride) % (N * M);
  int k = index_in_outer_dft % N;  // index in the contiguous factor/fft
  int n = index_in_outer_dft / N;  // index of the contiguous factor/fft
  int fft_start = local_id - index_in_outer_dft * stride;
  int source_wi_id = fft_start + stride * (k * M + n);
  real = sycl::select_from_group(sg, real, source_wi_id);
  imag = sycl::select_from_group(sg, imag, source_wi_id);
}

/**
 * Calculates DFT using Cooley-Tukey FFT algorithm. Size of the problem is N*M.
 * Each workitem holds one input and one output complex value.
 *
 * @tparam dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam N the first factor of the problem size
 * @tparam M the second factor of the problem size
 * @tparam stride stride between workitems working on consecutive values of one
 * DFT
 * @tparam T type of the scalar to work on
 * @param[in,out] real real component of the input/output complex value for one
 * workitem
 * @param[in,out] imag imaginary component of the input/output complex value for
 * one workitem
 * @param sg subgroup
 */
template <direction dir, int N, int M, int stride, typename T>
__attribute__((always_inline))  __attribute__((flatten)) inline void cross_sg_cooley_tukey_dft(T& real, T& imag, sycl::sub_group& sg) {
  int local_id = sg.get_local_linear_id();
  int index_in_outer_dft = (local_id / stride) % (N * M);
  int k = index_in_outer_dft % N;  // index in the contiguous factor/fft
  int n = index_in_outer_dft / N;  // index of the contiguous factor/fft

  // factor N
  cross_sg_dft<dir, N, M * stride>(real, imag, sg);
  // transpose
  cross_sg_transpose<N, M, stride>(real, imag, sg);
  // twiddle
  const T multi_re = twiddle<T>::re[N * M][k * n];
  const T multi_im = [&]() __attribute__((always_inline))  __attribute__((flatten)) {
    if constexpr (dir == direction::FORWARD) return twiddle<T>::im[N * M][k * n];
    return -twiddle<T>::im[N * M][k * n];
  }();
  T tmp_real = real * multi_re - imag * multi_im;
  imag = real * multi_im + imag * multi_re;
  real = tmp_real;
  // factor M
  cross_sg_dft<dir, M, N * stride>(real, imag, sg);
}

/**
 * Calculates DFT using FFT algorithm. Each workitem holds one input and one
 * output complex value.
 *
 * @tparam dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam N Size of the DFT
 * @tparam stride stride between workitems working on consecutive values of one
 * DFT
 * @tparam T type of the scalar to work on
 * @param[in,out] real real component of the input/output complex value for one
 * workitem
 * @param[in,out] imag imaginary component of the input/output complex value for
 * one workitem
 * @param sg subgroup
 */
template <direction dir, int N, int stride, typename T>
__attribute__((always_inline))  __attribute__((flatten)) inline void cross_sg_dft(T& real, T& imag, sycl::sub_group& sg) {
  constexpr int F0 = detail::factorize(N);
  if constexpr (F0 >= 2 && N / F0 >= 2) {
    cross_sg_cooley_tukey_dft<dir, N / F0, F0, stride>(real, imag, sg);
  } else {
    cross_sg_naive_dft<dir, N, stride>(real, imag, sg);
  }
}

/**
 * Factorizes a number into two factors, so that one of them will maximal below
 or equal to subgroup size.

 * @param N the number to factorize
 * @param sg_size subgroup size
 * @return the factor below or equal to subgroup size
 */
int factorize_sg(int N, int sg_size) {
  for (int i = sg_size; i > 1; i--) {
    if (N % i == 0) {
      return i;
    }
  }
  return 1;
}

};  // namespace detail

template <direction dir, int M, int N, typename T_ptr, typename T_twiddles_ptr>
__attribute__((always_inline)) inline void cross_sg_impl(T_ptr inout, sycl::sub_group& sg, T_twiddles_ptr sg_twiddles) {
   using T = detail::remove_ptr<T_ptr>;
  int idx_of_wi_in_fft = sg.get_local_linear_id() % N;

  detail::unrolled_loop<0, M, 1>([&](int idx_of_element_in_wi) __attribute__((always_inline))  __attribute__((flatten)) {
    T& real = inout[2 * idx_of_element_in_wi];
    T& imag = inout[2 * idx_of_element_in_wi + 1];

    detail::cross_sg_dft<dir, N, 1>(real, imag, sg);

    T twiddle_real = sg_twiddles[idx_of_element_in_wi * N + idx_of_wi_in_fft];
    T twiddle_imag = sg_twiddles[(idx_of_element_in_wi + M) * N + idx_of_wi_in_fft];
    if constexpr (dir == direction::BACKWARD) twiddle_imag = -twiddle_imag;
    T tmp_real = real * twiddle_real - imag * twiddle_imag;
    imag = real * twiddle_imag + imag * twiddle_real;
    real = tmp_real;
  });
}

/**
 * Calculates FFT of size N*M using workitems in a subgroup. Works in place. The
 * end result needs to be transposed when storing it to the local memory!
 *
 * @tparam dir direction of the FFT
 * @tparam M number of elements per workitem
 * @tparam T_prt type of the pointer to the data
 * @param N number of workitems in a subgroup that work on one FFT
 * @param inout pointer to private memory where the input/output data is
 * @param sg subgroup
 * @param sg_twiddles twiddle factors to use - calculated by sg_calc_twiddles in
 * commit
 */
template <direction dir, int M, int N, typename T_ptr, typename T_twiddles_ptr>
__attribute__((always_inline)) inline void sg_dft(T_ptr inout, sycl::sub_group& sg, T_twiddles_ptr sg_twiddles) {
  cross_sg_impl<dir, M, N>(inout, sg, sg_twiddles);
  wi_dft<dir, M, 1, 1>(inout, inout);
}

/**
 * Calculates a twiddle factor for subgroup implementation.
 *
 * @tparam T_ptr type of the pointer of accessor into which to store the
 * twiddles
 * @tparam N number of workitems in a subgroup that work on one FFT
 * @tparam M number of elements per workitem
 * @param n index of the twiddle to calculate in the direction of N
 * @param k index of the twiddle to calculate in the direction of M
 * @param sg_twiddles destination into which to store the twiddles
 */
template <typename T_ptr>
void sg_calc_twiddles(int N, int M, int n, int k, T_ptr sg_twiddles) {
  using T = detail::remove_ptr<T_ptr>;
  std::complex<T> twiddle = detail::calculate_twiddle<T>(n * k, N * M);
  sg_twiddles[k * N + n] = twiddle.real();
  sg_twiddles[(k + M) * N + n] = twiddle.imag();
}

};  // namespace sycl_fft

#endif