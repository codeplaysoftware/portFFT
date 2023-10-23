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

#ifndef PORTFFT_COMMON_WORKITEM_HPP
#define PORTFFT_COMMON_WORKITEM_HPP

#include <common/helpers.hpp>
#include <common/twiddle.hpp>
#include <defines.hpp>
#include <enums.hpp>
#include <sycl/sycl.hpp>

namespace portfft {

// forward declaration
template <direction Dir, Idx RecursionLevel, typename T>
inline void wi_dft(const T* in, T* out, Idx fft_size, Idx stride_in, Idx stride_out);

namespace detail {

// Maximum size of an FFT that can fit in the workitem implementation
static constexpr Idx MaxFftSizeWi = 56;

/*
`wi_dft` calculates a DFT by a workitem on values that are already loaded into its private memory.
It calls either `cooley_tukey_dft` (for composite sizes) or `naive_dft` (for prime sizes).

`cooley_tukey_dft` calculates DFT of a composite size by one workitem. It calls `wi_dft` for each of the factors and
does twiddle multiplication in-between. Transposition is handled by calling `wi_dft` with different input and output
strides.

`naive_dft` calculates DFT by one workitem using naive DFT algorithm.
*/

/**
 * Calculates DFT using naive algorithm. Can work in or out of place.
 *
 * @tparam Dir direction of the FFT
 * @tparam N size of the DFT transform
 * @tparam StrideIn stride (in complex values) between complex values in `in`
 * @tparam StrideOut stride (in complex values) between complex values in `out`
 * @tparam T type of the scalar used for computations
 * @param in pointer to input
 * @param out pointer to output
 */
template <direction Dir, typename T>
__attribute__((always_inline)) inline void naive_dft(const T* in, T* out, Idx fft_size, Idx stride_in, Idx stride_out) {
  std::array<T, 2 * MAX_COMPLEX_PER_WI> tmp;
  PORTFFT_UNROLL
  for (Idx idx_out = 0; idx_out < fft_size; idx_out++) {
    tmp[2 * idx_out + 0] = 0;
    tmp[2 * idx_out + 1] = 0;
    PORTFFT_UNROLL
    for (Idx idx_in = 0; idx_in < fft_size; idx_in++) {
      auto re_multiplier = twiddle<T>::Re[fft_size][idx_in * idx_out % fft_size];
      auto im_multiplier = [&]() {
        if constexpr (Dir == direction::FORWARD) {
          return twiddle<T>::Im[fft_size][idx_in * idx_out % fft_size];
        }
        return -twiddle<T>::Im[fft_size][idx_in * idx_out % fft_size];
      }();
      // multiply in and multi
      T tmp_real;
      T tmp_complex;
      detail::multiply_complex(in[2 * idx_in * stride_in], in[2 * idx_in * stride_in + 1], re_multiplier, im_multiplier,
                               tmp_real, tmp_complex);
      tmp[2 * idx_out + 0] += tmp_real;
      tmp[2 * idx_out + 1] += tmp_complex;
    }
  }
  PORTFFT_UNROLL
  for (Idx idx_out = 0; idx_out < 2 * fft_size; idx_out += 2) {
    out[idx_out * stride_out + 0] = tmp[idx_out + 0];
    out[idx_out * stride_out + 1] = tmp[idx_out + 1];
  }
}

// mem requirement: ~N*M(if in place, otherwise x2) + N*M(=tmp) + sqrt(N*M) + pow(N*M,0.25) + ...
// TODO explore if this tmp can be reduced/eliminated ^^^^^^
/**
 * Calculates DFT using Cooley-Tukey FFT algorithm. Can work in or out of place. Size of the problem is N*M
 *
 * @tparam Dir direction of the FFT
 * @tparam N the first factor of the problem size
 * @tparam M the second factor of the problem size
 * @tparam StrideIn stride (in complex values) between complex values in `in`
 * @tparam StrideOut stride (in complex values) between complex values in `out`
 * @tparam T type of the scalar used for computations
 * @param in pointer to input
 * @param out pointer to output
 */
template <direction Dir, Idx RecursionLevel, typename T>
__attribute__((always_inline)) inline void cooley_tukey_dft(const T* in, T* out, Idx N, Idx M, Idx stride_in,
                                                            Idx stride_out) {
  T tmp_buffer[MAX_COMPLEX_PER_WI];
  PORTFFT_UNROLL
  for (Idx i = 0; i < M; i++) {
    wi_dft<Dir, RecursionLevel>(in + 2 * i * stride_in, tmp_buffer + 2 * i * N, N, M * stride_in, 1);
    PORTFFT_UNROLL
    for (Idx j = 0; j < N; j++) {
      auto re_multiplier = twiddle<T>::Re[N * M][i * j];
      auto im_multiplier = [&]() {
        if constexpr (Dir == direction::FORWARD) {
          return twiddle<T>::Im[N * M][i * j];
        }
        return -twiddle<T>::Im[N * M][i * j];
      }();
      detail::multiply_complex(tmp_buffer[2 * i * N + 2 * j], tmp_buffer[2 * i * N + 2 * j + 1], re_multiplier,
                               im_multiplier, tmp_buffer[2 * i * N + 2 * j], tmp_buffer[2 * i * N + 2 * j + 1]);
    };
  }
  PORTFFT_UNROLL
  for (Idx i = 0; i < N; i++) {
    wi_dft<Dir, RecursionLevel>(tmp_buffer + 2 * i, out + 2 * i * stride_out, M, N, N * stride_out);
  };
}

/**
 * Factorizes a number into two roughly equal factors.
 * @tparam T type of the number to factorize
 * @param N the number to factorize
 * @return the smaller of the factors
 */
template <typename T>
constexpr T factorize(T N) {
  T res = 1;
  for (T i = 2; i * i <= N; i++) {
    if (N % i == 0) {
      res = i;
    }
  }
  return res;
}

/**
 * Calculates how many temporary complex values a workitem implementation needs
 * for solving FFT.
 * @tparam TIdx type of the size
 * @param N size of the FFT problem
 * @return Number of temporary complex values
 */
template <typename TIdx>
constexpr TIdx wi_temps(TIdx N) {
  TIdx f0 = factorize(N);
  TIdx f1 = N / f0;
  if (f0 < 2 || f1 < 2) {
    return N;
  }
  TIdx a = wi_temps(f0);
  TIdx b = wi_temps(f1);
  return (a > b ? a : b) + N;
}

/**
 * Checks whether a problem can be solved with workitem implementation without
 * registers spilling.
 * @tparam Scalar type of the real scalar used for the computation
 * @tparam TIdx type of the size
 * @param N Size of the problem, in complex values
 * @return true if the problem fits in the registers
 */
template <typename Scalar, typename TIdx>
constexpr bool fits_in_wi(TIdx N) {
  TIdx n_complex = N + wi_temps(N);
  TIdx complex_size = 2 * sizeof(Scalar);
  TIdx register_space = PORTFFT_REGISTERS_PER_WI * 4;
  return n_complex * complex_size <= register_space;
}

};  // namespace detail

/**
 * Calculates DFT using FFT algorithm. Can work in or out of place.
 *
 * @tparam Dir direction of the FFT
 * @tparam N size of the DFT transform
 * @tparam StrideIn stride (in complex values) between complex values in `in`
 * @tparam StrideOut stride (in complex values) between complex values in `out`
 * @tparam T type of the scalar used for computations
 * @param in pointer to input
 * @param out pointer to output
 */
template <direction Dir, Idx RecursionLevel, typename T>
__attribute__((always_inline)) inline void wi_dft(const T* in, T* out, Idx fft_size, Idx stride_in, Idx stride_out) {
  const Idx F0 = detail::factorize(fft_size);
  constexpr Idx MaxRecursionLevel = detail::uint_log2(MAX_COMPLEX_PER_WI) - 1;
  if constexpr (RecursionLevel < MaxRecursionLevel) {
    if (fft_size == 2) {
      T a = in[0 * stride_in + 0] + in[2 * stride_in + 0];
      T b = in[0 * stride_in + 1] + in[2 * stride_in + 1];
      T c = in[0 * stride_in + 0] - in[2 * stride_in + 0];
      out[2 * stride_out + 1] = in[0 * stride_in + 1] - in[2 * stride_in + 1];
      out[0 * stride_out + 0] = a;
      out[0 * stride_out + 1] = b;
      out[2 * stride_out + 0] = c;
    } else if (F0 >= 2 && fft_size / F0 >= 2) {
      detail::cooley_tukey_dft<Dir, RecursionLevel + 1>(in, out, fft_size / F0, F0, stride_in, stride_out);
    } else {
      detail::naive_dft<Dir>(in, out, fft_size, stride_in, stride_out);
    }
  }
}

};  // namespace portfft

#endif
