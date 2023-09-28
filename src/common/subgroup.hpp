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
 *  Codeplay's portFFT
 *
 **************************************************************************/

#ifndef PORTFFT_COMMON_SUBGROUP_HPP
#define PORTFFT_COMMON_SUBGROUP_HPP

#include "helpers.hpp"
#include <common/helpers.hpp>
#include <common/twiddle.hpp>
#include <common/twiddle_calc.hpp>
#include <common/workitem.hpp>
#include <enums.hpp>
#include <sycl/sycl.hpp>

namespace portfft {
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

`cross_sg_dft` calculates DFT across workitems, with each workitem contributing one complex value as input and output of
the computation. If the size of the subgroup is large enough compared to FFT size, a subgroup can calculate multiple
DFTs at once (the same holds true for `cross_sg_cooley_tukey_dft` and `cross_sg_naive_dft`). It calls either
`cross_sg_cooley_tukey_dft` (for composite sizes) or `cross_sg_naive_dft` (for prime sizes).

`cross_sg_cooley_tukey_dft` calculates DFT of a composite size across workitems. It calls `cross_sg_dft` for each of the
factors and does transposition and twiddle multiplication inbetween.

`cross_sg_naive_dft` calculates DFT across workitems using naive DFT algorithm.
*/

// forward declaration
template <direction Dir, int N, int Stride, typename T>
inline void cross_sg_dft(T& real, T& imag, sycl::sub_group& sg);

/**
 * Calculates DFT using naive algorithm by using workitems of one subgroup.
 * Each workitem holds one input and one output complex value.
 *
 * @tparam Dir direction of the FFT
 * @tparam N size of the DFT transform
 * @tparam Stride Stride between workitems working on consecutive values of one
 * DFT
 * @tparam T type of the scalar to work on
 * @param[in,out] real real component of the input/output complex value for one
 * workitem
 * @param[in,out] imag imaginary component of the input/output complex value for
 * one workitem
 * @param sg subgroup
 */
template <direction Dir, int N, int Stride, typename T>
__attribute__((always_inline)) inline void cross_sg_naive_dft(T& real, T& imag, sycl::sub_group& sg) {
  if constexpr (N == 2 && (Stride & (Stride - 1)) == 0) {
    int local_id = static_cast<int>(sg.get_local_linear_id());
    int idx_out = (local_id / Stride) % 2;

    T multi_re = (idx_out & 1) ? T(-1) : T(1);
    T res_real = real * multi_re;
    T res_imag = imag * multi_re;

    res_real += sycl::permute_group_by_xor(sg, real, Stride);
    res_imag += sycl::permute_group_by_xor(sg, imag, Stride);

    real = res_real;
    imag = res_imag;
  } else {
    int local_id = static_cast<int>(sg.get_local_linear_id());
    int idx_out = (local_id / Stride) % N;
    int fft_start = local_id - idx_out * Stride;

    T res_real = 0;
    T res_imag = 0;

    unrolled_loop<0, N, 1>([&](int idx_in) __attribute__((always_inline)) {
      const T multi_re = twiddle<T>::Re[N][idx_in * idx_out % N];
      const T multi_im = [&]() __attribute__((always_inline)) {
        if constexpr (Dir == direction::FORWARD) {
          return twiddle<T>::Im[N][idx_in * idx_out % N];
        }
        return -twiddle<T>::Im[N][idx_in * idx_out % N];
      }
      ();
      std::size_t source_wi_id = static_cast<std::size_t>(fft_start + idx_in * Stride);

      T cur_real = sycl::select_from_group(sg, real, source_wi_id);
      T cur_imag = sycl::select_from_group(sg, imag, source_wi_id);

      // multiply cur and multi
      T tmp_real;
      T tmp_imag;
      multiply_complex(static_cast<const T>(cur_real), static_cast<const T>(cur_imag), static_cast<const T>(multi_re),
                       static_cast<const T>(multi_im), tmp_real, tmp_imag);
      res_real += tmp_real;
      res_imag += tmp_imag;
    });

    real = res_real;
    imag = res_imag;
  }
}

/**
 * Transposes values held by workitems of a subgroup. Transposes rectangles of
 * size N*M. Each of the rectangles can be strided.
 *
 * @tparam N inner - contiguous size on input, outer size on output
 * @tparam M outer size on input, inner - contiguous size on output
 * @tparam Stride Stride between consecutive values of one rectangle
 * @tparam T type of the scalar to work on
 * @param[in,out] real real component of the input/output complex value for one
 * workitem
 * @param[in,out] imag imaginary component of the input/output complex value for
 * one workitem
 * @param sg subgroup
 */
template <int N, int M, int Stride, typename T>
__attribute__((always_inline)) inline void cross_sg_transpose(T& real, T& imag, sycl::sub_group& sg) {
  int local_id = static_cast<int>(sg.get_local_linear_id());
  int index_in_outer_dft = (local_id / Stride) % (N * M);
  int k = index_in_outer_dft % N;  // index in the contiguous factor/fft
  int n = index_in_outer_dft / N;  // index of the contiguous factor/fft
  int fft_start = local_id - index_in_outer_dft * Stride;
  int source_wi_id = fft_start + Stride * (k * M + n);
  real = sycl::select_from_group(sg, real, static_cast<std::size_t>(source_wi_id));
  imag = sycl::select_from_group(sg, imag, static_cast<std::size_t>(source_wi_id));
}

/**
 * Calculates DFT using Cooley-Tukey FFT algorithm. Size of the problem is N*M.
 * Each workitem holds one input and one output complex value.
 *
 * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam N the first factor of the problem size
 * @tparam M the second factor of the problem size
 * @tparam Stride Stride between workitems working on consecutive values of one
 * DFT
 * @tparam T type of the scalar to work on
 * @param[in,out] real real component of the input/output complex value for one
 * workitem
 * @param[in,out] imag imaginary component of the input/output complex value for
 * one workitem
 * @param sg subgroup
 */
template <direction Dir, int N, int M, int Stride, typename T>
__attribute__((always_inline)) inline void cross_sg_cooley_tukey_dft(T& real, T& imag, sycl::sub_group& sg) {
  int local_id = static_cast<int>(sg.get_local_linear_id());
  int index_in_outer_dft = (local_id / Stride) % (N * M);
  int k = index_in_outer_dft % N;  // index in the contiguous factor/fft
  int n = index_in_outer_dft / N;  // index of the contiguous factor/fft

  // factor N
  cross_sg_dft<Dir, N, M * Stride>(real, imag, sg);
  // transpose
  cross_sg_transpose<N, M, Stride>(real, imag, sg);
  // twiddle
  const T multi_re = twiddle<T>::Re[N * M][k * n];
  const T multi_im = [&]() __attribute__((always_inline)) {
    if constexpr (Dir == direction::FORWARD) {
      return twiddle<T>::Im[N * M][k * n];
    }
    return -twiddle<T>::Im[N * M][k * n];
  }
  ();
  multiply_complex(static_cast<const T>(real), static_cast<const T>(imag), multi_re, multi_im, real, imag);
  // factor M
  cross_sg_dft<Dir, M, N * Stride>(real, imag, sg);
}

/**
 * Calculates DFT using FFT algorithm. Each workitem holds one input and one
 * output complex value.
 *
 * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam N Size of the DFT
 * @tparam Stride Stride between workitems working on consecutive values of one
 * DFT
 * @tparam T type of the scalar to work on
 * @param[in,out] real real component of the input/output complex value for one
 * workitem
 * @param[in,out] imag imaginary component of the input/output complex value for
 * one workitem
 * @param sg subgroup
 */
template <direction Dir, int N, int Stride, typename T>
__attribute__((always_inline)) inline void cross_sg_dft(T& real, T& imag, sycl::sub_group& sg) {
  constexpr int F0 = detail::factorize(N);
  if constexpr (F0 >= 2 && N / F0 >= 2) {
    cross_sg_cooley_tukey_dft<Dir, N / F0, F0, Stride>(real, imag, sg);
  } else {
    cross_sg_naive_dft<Dir, N, Stride>(real, imag, sg);
  }
}

/**
 * Factorizes a number into two factors, so that one of them will maximal below
 or equal to subgroup size.
 * @param N the number to factorize
 * @param sg_size subgroup size
 * @return the factor below or equal to subgroup size
 */
constexpr int factorize_sg(int N, int sg_size) {
  if constexpr (PORTFFT_SLOW_SG_SHUFFLES) {
    return 1;
  } else {
    for (int i = sg_size; i > 1; i--) {
      if (N % i == 0) {
        return i;
      }
    }
    return 1;
  }
}

/**
 * Checks whether a problem can be solved with sub-group implementation
 * without reg spilling.
 * @tparam Scalar type of the real scalar used for the computation
 * @tparam TIndex Index type
 * @param N Size of the problem, in complex values
 * @param sg_size Size of the sub-group
 * @return true if the problem fits in the registers
 */
template <typename Scalar, typename TIndex>
constexpr bool fits_in_sg(TIndex N, int sg_size) {
  int factor_sg = factorize_sg(static_cast<int>(N), sg_size);
  int factor_wi = static_cast<int>(N) / factor_sg;
  return fits_in_wi<Scalar>(factor_wi);
}

};  // namespace detail

/**
 * Calculates FFT of size N*M using workitems in a subgroup. Works in place. The
 * end result needs to be transposed when storing it to the local memory!
 *
 * @tparam Dir direction of the FFT
 * @tparam M number of elements per workitem
 * @tparam N number of workitems in a subgroup that work on one FFT
 * @tparam T type of the scalar used for computations
 * @param inout pointer to private memory where the input/output data is
 * @param sg subgroup
 * @param sg_twiddles twiddle factors to use - calculated by sg_calc_twiddles in
 * commit
 */
template <direction Dir, int M, int N, typename T>
__attribute__((always_inline)) inline void sg_dft(T* inout, sycl::sub_group& sg, const T* sg_twiddles) {
  int idx_of_wi_in_fft = static_cast<int>(sg.get_local_linear_id()) % N;

  detail::unrolled_loop<0, M, 1>([&](int idx_of_element_in_wi) __attribute__((always_inline)) {
    T& real = inout[2 * idx_of_element_in_wi];
    T& imag = inout[2 * idx_of_element_in_wi + 1];

    if constexpr (N > 1) {
      detail::cross_sg_dft<Dir, N, 1>(real, imag, sg);
      if (idx_of_element_in_wi > 0) {
        T twiddle_real = sg_twiddles[idx_of_element_in_wi * N + idx_of_wi_in_fft];
        T twiddle_imag = sg_twiddles[(idx_of_element_in_wi + M) * N + idx_of_wi_in_fft];
        if constexpr (Dir == direction::BACKWARD) {
          twiddle_imag = -twiddle_imag;
        }
        detail::multiply_complex(static_cast<const T>(real), static_cast<const T>(imag),
                                 static_cast<const T>(twiddle_real), static_cast<const T>(twiddle_imag), real, imag);
      }
    }
  });

  wi_dft<Dir, M, 1, 1>(inout, inout);
}

/**
 * Calculates a twiddle factor for subgroup implementation.
 *
 * @tparam T type of the scalar used for computations
 * @param N number of workitems in a subgroup that work on one FFT
 * @param M number of elements per workitem
 * @param n index of the twiddle to calculate in the direction of N
 * @param k index of the twiddle to calculate in the direction of M
 * @param sg_twiddles destination into which to store the twiddles
 */
template <typename T>
void sg_calc_twiddles(int N, int M, int n, int k, T* sg_twiddles) {
  std::complex<T> twiddle = detail::calculate_twiddle<T>(n * k, N * M);
  sg_twiddles[k * N + n] = twiddle.real();
  sg_twiddles[(k + M) * N + n] = twiddle.imag();
}

};  // namespace portfft

#endif
