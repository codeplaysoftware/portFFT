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
template <direction Dir, int SubgroupSize, int XSgDftRecursionLevel, typename T>
inline void cross_sg_dft(std::size_t dft_size, std::size_t stride, T& real, T& imag, sycl::sub_group& sg);

/**
 * Calculates DFT using naive algorithm by using workitems of one subgroup.
 * Each workitem holds one input and one output complex value.
 *
 * @tparam Dir direction of the FFT
 * @tparam T type of the scalar to work on
 * @param dft_size size of the DFT transform
 * @param stride stride between workitems working on consecutive values of one
 * DFT
 * @param[in,out] real real component of the input/output complex value for one
 * workitem
 * @param[in,out] imag imaginary component of the input/output complex value for
 * one workitem
 * @param sg subgroup
 */
template <direction Dir, typename T>
__attribute__((always_inline)) inline void cross_sg_naive_dft(std::size_t dft_size, std::size_t stride, T& real,
                                                              T& imag, sycl::sub_group& sg) {
  if (dft_size == 2 && (stride & (stride - 1)) == 0) {
    std::size_t local_id = sg.get_local_linear_id();
    std::size_t idx_out = (local_id / stride) % 2;

    T multi_re = (idx_out & 1) ? T(-1) : T(1);
    T res_real = real * multi_re;
    T res_imag = imag * multi_re;

    res_real += sycl::permute_group_by_xor(sg, real, static_cast<typename sycl::sub_group::linear_id_type>(stride));
    res_imag += sycl::permute_group_by_xor(sg, imag, static_cast<typename sycl::sub_group::linear_id_type>(stride));

    real = res_real;
    imag = res_imag;
  } else {
    std::size_t local_id = sg.get_local_linear_id();
    std::size_t idx_out = (local_id / stride) % dft_size;
    std::size_t fft_start = local_id - idx_out * stride;

    T res_real = 0;
    T res_imag = 0;

    for (std::size_t idx_in{0}; idx_in < dft_size; ++idx_in) {
      const T multi_re = twiddle<T>::Re[dft_size][idx_in * idx_out % dft_size];
      const T multi_im = [&]() __attribute__((always_inline)) {
        if constexpr (Dir == direction::FORWARD) {
          return twiddle<T>::Im[dft_size][idx_in * idx_out % dft_size];
        }
        return -twiddle<T>::Im[dft_size][idx_in * idx_out % dft_size];
      }
      ();
      std::size_t source_wi_id = fft_start + idx_in * stride;

      T cur_real = sycl::select_from_group(sg, real, source_wi_id);
      T cur_imag = sycl::select_from_group(sg, imag, source_wi_id);

      // multiply cur and multi
      res_real += cur_real * multi_re - cur_imag * multi_im;
      res_imag += cur_real * multi_im + cur_imag * multi_re;
    }

    real = res_real;
    imag = res_imag;
  }
}

/**
 * Transposes values held by workitems of a subgroup. Transposes rectangles of
 * size factor_n*factor_m. Each of the rectangles can be strided.
 *
 * @tparam T type of the scalar to work on
 * @param factor_n inner - contiguous size on input, outer size on output
 * @param factor_m outer size on input, inner - contiguous size on output
 * @param stride stride between consecutive values of one rectangle
 * @param[in,out] real real component of the input/output complex value for one
 * workitem
 * @param[in,out] imag imaginary component of the input/output complex value for
 * one workitem
 * @param sg subgroup
 */
template <typename T>
__attribute__((always_inline)) inline void cross_sg_transpose(std::size_t factor_n, std::size_t factor_m,
                                                              std::size_t stride, T& real, T& imag,
                                                              sycl::sub_group& sg) {
  std::size_t local_id = sg.get_local_linear_id();
  std::size_t index_in_outer_dft = (local_id / stride) % (factor_n * factor_m);
  std::size_t k = index_in_outer_dft % factor_n;  // index in the contiguous factor/fft
  std::size_t n = index_in_outer_dft / factor_n;  // index of the contiguous factor/fft
  std::size_t fft_start = local_id - index_in_outer_dft * stride;
  std::size_t source_wi_id = fft_start + stride * (k * factor_m + n);
  real = sycl::select_from_group(sg, real, source_wi_id);
  imag = sycl::select_from_group(sg, imag, source_wi_id);
}

/**
 * Calculates DFT using Cooley-Tukey FFT algorithm. Size of the problem is factor_n*factor_m.
 * Each workitem holds one input and one output complex value.
 *
 * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam XSgDftRecursionLevel The number of times cross_sg_dft has been recursively called prior to calling this
 * function.
 * @tparam T type of the scalar to work on
 * @param factor_n the first factor of the problem size
 * @param factor_m the second factor of the problem size
 * @param stride stride between workitems working on consecutive values of one
 * DFT
 * @param[in,out] real real component of the input/output complex value for one
 * workitem
 * @param[in,out] imag imaginary component of the input/output complex value for
 * one workitem
 * @param sg subgroup
 */
template <direction Dir, int SubgroupSize, int XSgDftRecursionLevel, typename T>
__attribute__((always_inline)) inline void cross_sg_cooley_tukey_dft(std::size_t factor_n, std::size_t factor_m,
                                                                     std::size_t stride, T& real, T& imag,
                                                                     sycl::sub_group& sg) {
  std::size_t local_id = sg.get_local_linear_id();
  std::size_t index_in_outer_dft = (local_id / stride) % (factor_n * factor_m);
  std::size_t k = index_in_outer_dft % factor_n;  // index in the contiguous factor/fft
  std::size_t n = index_in_outer_dft / factor_n;  // index of the contiguous factor/fft

  // factor N
  cross_sg_dft<Dir, SubgroupSize, XSgDftRecursionLevel>(factor_n, factor_m * stride, real, imag, sg);
  // transpose
  cross_sg_transpose(factor_n, factor_m, stride, real, imag, sg);
  // twiddle
  const T multi_re = twiddle<T>::Re[factor_n * factor_m][k * n];
  const T multi_im = [&]() __attribute__((always_inline)) {
    if constexpr (Dir == direction::FORWARD) {
      return twiddle<T>::Im[factor_n * factor_m][k * n];
    }
    return -twiddle<T>::Im[factor_n * factor_m][k * n];
  }
  ();
  T tmp_real = real * multi_re - imag * multi_im;
  imag = real * multi_im + imag * multi_re;
  real = tmp_real;
  // factor M
  cross_sg_dft<Dir, SubgroupSize, XSgDftRecursionLevel>(factor_m, factor_n * stride, real, imag, sg);
}

/**
 * Calculates DFT using FFT algorithm. Each workitem holds one input and one
 * output complex value.
 *
 * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam XSgDftRecursionLevel The number of times cross_sg_dft has been recursively called prior to calling this
 * function.
 * @tparam T type of the scalar to work on
 * @param dft_size Size of the DFT
 * @param stride Stride between workitems working on consecutive values of one
 * DFT
 * @param[in,out] real real component of the input/output complex value for one
 * workitem
 * @param[in,out] imag imaginary component of the input/output complex value for
 * one workitem
 * @param sg subgroup
 */
template <direction Dir, int SubgroupSize, int XSgDftRecursionLevel, typename T>
__attribute__((always_inline)) inline void cross_sg_dft(std::size_t dft_size, std::size_t stride, T& real, T& imag,
                                                        sycl::sub_group& sg) {
  // Max DFT size is sub-group size.
  constexpr int MaxXSgDftRecursionLevel = detail::uint_log2(SubgroupSize);
  if constexpr (XSgDftRecursionLevel < MaxXSgDftRecursionLevel) {
    std::size_t f0 = detail::factorize(dft_size);
    if (f0 >= 2 && dft_size / f0 >= 2) {
      cross_sg_cooley_tukey_dft<Dir, SubgroupSize, XSgDftRecursionLevel + 1>(dft_size / f0, f0, stride, real, imag, sg);
    } else {
      cross_sg_naive_dft<Dir>(dft_size, stride, real, imag, sg);
    }
  }
}

/**
 * Factorizes a number into two factors, so that one of them will maximal below
 or equal to subgroup size.
 * @tparam IntT The integer type to use for N and to return.
 * @param N the number to factorize
 * @param sg_size subgroup size
 * @return the factor below or equal to subgroup size
 */
template <typename IntT>
__attribute__((always_inline)) constexpr IntT factorize_sg(IntT N, int sg_size) {
  if constexpr (PORTFFT_SLOW_SG_SHUFFLES) {
    return 1;
  } else {
    for (IntT i = static_cast<IntT>(sg_size); i > 1; i--) {
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
__attribute__((always_inline)) constexpr bool fits_in_sg(TIndex N, int sg_size) {
  int factor_sg = factorize_sg(static_cast<int>(N), sg_size);
  int factor_wi = static_cast<int>(N) / factor_sg;
  return fits_in_wi<Scalar>(factor_wi);
}

};  // namespace detail

/**
 * Calculates FFT of size factor_n*factor_m using workitems in a subgroup. Works in place. The
 * end result needs to be transposed when storing it to the local memory!
 *
 * @tparam Dir direction of the FFT
 * @tparam T type of the scalar used for computations
 * @param factor_m number of elements per workitem
 * @param factor_n number of workitems in a subgroup that work on one FFT
 * @param inout pointer to private memory where the input/output data is
 * @param sg subgroup
 * @param sg_twiddles twiddle factors to use - calculated by sg_calc_twiddles in
 * commit
 * @param wi_private_scratch Scratch memory for this WI in WI impl.
 */
template <direction Dir, int SubgroupSize, typename T>
__attribute__((always_inline)) inline void sg_dft(std::size_t factor_m, std::size_t factor_n, T* inout,
                                                  sycl::sub_group& sg, const T* sg_twiddles, T* wi_private_scratch) {
  std::size_t idx_of_wi_in_fft = sg.get_local_linear_id() % factor_n;

  for (std::size_t idx_of_element_in_wi{0}; idx_of_element_in_wi < factor_m; ++idx_of_element_in_wi) {
    T& real = inout[2 * idx_of_element_in_wi];
    T& imag = inout[2 * idx_of_element_in_wi + 1];

    if (factor_n > 1) {
      detail::cross_sg_dft<Dir, SubgroupSize, 0>(factor_n, 1, real, imag, sg);
      if (idx_of_element_in_wi > 0) {
        T twiddle_real = sg_twiddles[idx_of_element_in_wi * factor_n + idx_of_wi_in_fft];
        T twiddle_imag = sg_twiddles[(idx_of_element_in_wi + factor_m) * factor_n + idx_of_wi_in_fft];
        if constexpr (Dir == direction::BACKWARD) {
          twiddle_imag = -twiddle_imag;
        }
        T tmp_real = real * twiddle_real - imag * twiddle_imag;
        imag = real * twiddle_imag + imag * twiddle_real;
        real = tmp_real;
      }
    }
  }
  wi_dft<Dir, 0>(static_cast<int>(factor_m), inout, 1, inout, 1, wi_private_scratch);
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
__attribute__((always_inline)) void sg_calc_twiddles(int N, int M, int n, int k, T* sg_twiddles) {
  std::complex<T> twiddle = detail::calculate_twiddle<T>(n * k, N * M);
  sg_twiddles[k * N + n] = twiddle.real();
  sg_twiddles[(k + M) * N + n] = twiddle.imag();
}

};  // namespace portfft

#endif
