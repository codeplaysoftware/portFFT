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

#ifndef SYCL_FFT_COMMON_WORKITEM_HPP
#define SYCL_FFT_COMMON_WORKITEM_HPP

#include <common/helpers.hpp>
#include <common/twiddle.hpp>
#include <enums.hpp>
#include <sycl/sycl.hpp>

namespace sycl_fft {

// forward declaration
template <direction dir, auto N, auto stride_in, auto stride_out, typename T>
inline void wi_dft(const T* in, T* out);

namespace detail {

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
 * @tparam dir direction of the FFT
 * @tparam N size of the DFT transform
 * @tparam stride_in stride (in complex values) between complex values in `in`
 * @tparam stride_out stride (in complex values) between complex values in `out`
 * @tparam T type of the scalar used for computations
 * @param in pointer to input
 * @param out pointer to output
 */
template <direction dir, auto N, auto stride_in, auto stride_out, typename T>
__attribute__((always_inline)) inline void naive_dft(const T* in, T* out) {
  using T_index = decltype(N);
  T tmp[2 * N];
  unrolled_loop<0, N, 1, T_index>([&](T_index idx_out) __attribute__((always_inline)) {
    tmp[2 * idx_out + 0] = 0;
    tmp[2 * idx_out + 1] = 0;
    unrolled_loop<0, N, 1, T_index>([&](T_index idx_in) __attribute__((always_inline)) {
      // this multiplier is not really a twiddle factor, but it is calculated the same way
      auto re_multiplier = twiddle<T>::re[N][idx_in * idx_out % N];
      auto im_multiplier = [&]() {
        if constexpr (dir == direction::FORWARD) return twiddle<T>::im[N][idx_in * idx_out % N];
        return -twiddle<T>::im[N][idx_in * idx_out % N];
      }();

      // multiply in and multi
      tmp[2 * idx_out + 0] +=
          in[2 * idx_in * stride_in] * re_multiplier - in[2 * idx_in * stride_in + 1] * im_multiplier;
      tmp[2 * idx_out + 1] +=
          in[2 * idx_in * stride_in] * im_multiplier + in[2 * idx_in * stride_in + 1] * re_multiplier;
    });
  });
  unrolled_loop<0, 2 * N, 2, T_index>([&](T_index idx_out) {
    out[idx_out * stride_out + 0] = tmp[idx_out + 0];
    out[idx_out * stride_out + 1] = tmp[idx_out + 1];
  });
}

// mem requirement: ~N*M(if in place, otherwise x2) + N*M(=tmp) + sqrt(N*M) + pow(N*M,0.25) + ...
// TODO explore if this tmp can be reduced/eliminated ^^^^^^
/**
 * Calculates DFT using Cooley-Tukey FFT algorithm. Can work in or out of place. Size of the problem is N*M
 *
 * @tparam dir direction of the FFT
 * @tparam N the first factor of the problem size
 * @tparam M the second factor of the problem size
 * @tparam stride_in stride (in complex values) between complex values in `in`
 * @tparam stride_out stride (in complex values) between complex values in `out`
 * @tparam T type of the scalar used for computations
 * @param in pointer to input
 * @param out pointer to output
 */
template <direction dir, auto N, auto M, auto stride_in, auto stride_out, typename T>
__attribute__((always_inline)) inline void cooley_tukey_dft(const T* in, T* out) {
  using T_index = decltype(N);
  T tmp_buffer[2 * N * M];

  unrolled_loop<0, M, 1, T_index>([&](T_index i) __attribute__((always_inline)) {
    wi_dft<dir, N, M * stride_in, 1>(in + 2 * i * stride_in, tmp_buffer + 2 * i * N);
    unrolled_loop<0, N, 1, T_index>([&](T_index j) __attribute__((always_inline)) {
      auto re_multiplier = twiddle<T>::re[N * M][i * j];
      auto im_multiplier = [&]() {
        if constexpr (dir == direction::FORWARD) return twiddle<T>::im[N * M][i * j];
        return -twiddle<T>::im[N * M][i * j];
      }();
      T tmp_val = tmp_buffer[2 * i * N + 2 * j] * re_multiplier - tmp_buffer[2 * i * N + 2 * j + 1] * im_multiplier;
      tmp_buffer[2 * i * N + 2 * j + 1] =
          tmp_buffer[2 * i * N + 2 * j] * im_multiplier + tmp_buffer[2 * i * N + 2 * j + 1] * re_multiplier;
      tmp_buffer[2 * i * N + 2 * j + 0] = tmp_val;
    });
  });
  unrolled_loop<0, N, 1, T_index>([&](T_index i) __attribute__((always_inline)) {
    wi_dft<dir, M, N, N * stride_out>(tmp_buffer + 2 * i, out + 2 * i * stride_out);
  });
}

/**
 * Factorizes a number into two roughly equal factors.
 * @tparam T_index Index type
 * @param N the number to factorize
 * @return the smaller of the factors
 */
template <typename T_index>
constexpr T_index factorize(T_index N) {
  T_index res = 1;
  for (T_index i = 2; i * i <= N; i++) {
    if (N % i == 0) {
      res = i;
    }
  }
  return res;
}

/**
 * Calculates how many temporary complex values a workitem implementation needs
 * for solving FFT.
 * @tparam T_index Index type
 * @param N size of the FFT problem
 * @return Number of temporary complex values
 */
template <typename T_index>
constexpr T_index wi_temps(T_index N) {
  T_index F0 = factorize(N);
  T_index F1 = N / F0;
  if (F0 < 2 || F1 < 2) {
    return N;
  }
  T_index a = wi_temps(F0);
  T_index b = wi_temps(F1);
  return (a > b ? a : b) + N;
}

/**
 * Checks whether a problem can be solved with workitem implementation without
 * registers spilling.
 * @tparam Scalar type of the real scalar used for the computation
 * @tparam T_index Index type
 * @param N Size of the problem, in complex values
 * @return true if the problem fits in the registers
 */
template <typename Scalar, typename T_index>
constexpr bool fits_in_wi(T_index N) {
  T_index N_complex = N + wi_temps(N);
  T_index complex_size = 2 * sizeof(Scalar);
  T_index register_space = SYCLFFT_TARGET_REGS_PER_WI * 4;
  return N_complex * complex_size <= register_space;
}

/**
 * Struct with precalculated values for all relevant arguments to
 * fits_in_wi for use on device, where recursive functions are not allowed.
 *
 * @tparam Scalar type of the real scalar used for the computation
 */
template <typename Scalar>
struct fits_in_wi_device_struct {
  static constexpr bool buf[56] = {
      fits_in_wi<Scalar>(1),  fits_in_wi<Scalar>(2),  fits_in_wi<Scalar>(3),  fits_in_wi<Scalar>(4),
      fits_in_wi<Scalar>(5),  fits_in_wi<Scalar>(6),  fits_in_wi<Scalar>(7),  fits_in_wi<Scalar>(8),
      fits_in_wi<Scalar>(9),  fits_in_wi<Scalar>(10), fits_in_wi<Scalar>(11), fits_in_wi<Scalar>(12),
      fits_in_wi<Scalar>(13), fits_in_wi<Scalar>(14), fits_in_wi<Scalar>(15), fits_in_wi<Scalar>(16),
      fits_in_wi<Scalar>(17), fits_in_wi<Scalar>(18), fits_in_wi<Scalar>(19), fits_in_wi<Scalar>(20),
      fits_in_wi<Scalar>(21), fits_in_wi<Scalar>(22), fits_in_wi<Scalar>(23), fits_in_wi<Scalar>(24),
      fits_in_wi<Scalar>(25), fits_in_wi<Scalar>(26), fits_in_wi<Scalar>(27), fits_in_wi<Scalar>(28),
      fits_in_wi<Scalar>(29), fits_in_wi<Scalar>(30), fits_in_wi<Scalar>(31), fits_in_wi<Scalar>(32),
      fits_in_wi<Scalar>(33), fits_in_wi<Scalar>(34), fits_in_wi<Scalar>(35), fits_in_wi<Scalar>(36),
      fits_in_wi<Scalar>(37), fits_in_wi<Scalar>(38), fits_in_wi<Scalar>(39), fits_in_wi<Scalar>(40),
      fits_in_wi<Scalar>(41), fits_in_wi<Scalar>(42), fits_in_wi<Scalar>(43), fits_in_wi<Scalar>(44),
      fits_in_wi<Scalar>(45), fits_in_wi<Scalar>(46), fits_in_wi<Scalar>(47), fits_in_wi<Scalar>(48),
      fits_in_wi<Scalar>(49), fits_in_wi<Scalar>(50), fits_in_wi<Scalar>(51), fits_in_wi<Scalar>(52),
      fits_in_wi<Scalar>(53), fits_in_wi<Scalar>(54), fits_in_wi<Scalar>(55), fits_in_wi<Scalar>(56),
  };
};

/**
 * Checks whether a problem can be solved with workitem implementation without
 * registers spilling. Non-recursive implementation for the use on device.
 * @tparam Scalar type of the real scalar used for the computation
 * @tparam T_index Index type
 * @param N Size of the problem, in complex values
 * @return true if the problem fits in the registers
 */
template <typename Scalar, typename T_index>
__attribute__((always_inline)) inline bool fits_in_wi_device(T_index fft_size) {
  // 56 is the maximal size we support in workitem implementation and also
  // the size of the array above that is used if this if is not taken
  if (fft_size > 56) {
    return false;
  }
  return fits_in_wi_device_struct<Scalar>::buf[fft_size - 1];
}

};  // namespace detail

/**
 * Calculates DFT using FFT algorithm. Can work in or out of place.
 *
 * @tparam dir direction of the FFT
 * @tparam N size of the DFT transform
 * @tparam stride_in stride (in complex values) between complex values in `in`
 * @tparam stride_out stride (in complex values) between complex values in `out`
 * @tparam T type of the scalar used for computations
 * @param in pointer to input
 * @param out pointer to output
 */
template <direction dir, auto N, auto stride_in, auto stride_out, typename T>
__attribute__((always_inline)) inline void wi_dft(const T* in, T* out) {
  constexpr auto F0 = detail::factorize(N);
  if constexpr (N == 2) {
    T a = in[0 * stride_in + 0] + in[2 * stride_in + 0];
    T b = in[0 * stride_in + 1] + in[2 * stride_in + 1];
    T c = in[0 * stride_in + 0] - in[2 * stride_in + 0];
    out[2 * stride_out + 1] = in[0 * stride_in + 1] - in[2 * stride_in + 1];
    out[0 * stride_out + 0] = a;
    out[0 * stride_out + 1] = b;
    out[2 * stride_out + 0] = c;
  } else if constexpr (F0 >= 2 && N / F0 >= 2) {
    detail::cooley_tukey_dft<dir, N / F0, F0, stride_in, stride_out>(in, out);
  } else {
    detail::naive_dft<dir, N, stride_in, stride_out>(in, out);
  }
}

};  // namespace sycl_fft

#endif
