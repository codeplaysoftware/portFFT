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
template <direction dir, int N, int stride_in, int stride_out, typename T_ptr>
inline void wi_dft(T_ptr in, T_ptr out);

namespace detail {

/**
 * Calculates DFT using naive algorithm. Can work in or out of place.
 *
 * @tparam dir direction of the FFT
 * @tparam N size of the DFT transform
 * @tparam stride_in stride (in complex values) between complex values in `in`
 * @tparam stride_out stride (in complex values) between complex values in `out`
 * @tparam T_ptr type of pointer for `in` and `out`. Can be raw pointer or sycl::multi_ptr.
 * @param in pointer to input
 * @param out pointer to output
 */
template <direction dir, int N, int stride_in, int stride_out, typename T_ptr>
inline __attribute__((always_inline)) void naive_dft(T_ptr in, T_ptr out) {
  using T = remove_ptr<T_ptr>;
  constexpr T TWOPI = 2.0 * M_PI;
  T tmp[2 * N];
  unrolled_loop<0, N, 1>([&](int idx_out) __attribute__((always_inline)) {
    tmp[2 * idx_out + 0] = 0;
    tmp[2 * idx_out + 1] = 0;
    unrolled_loop<0, N, 1>([&](int idx_in) __attribute__((always_inline)) {
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
  unrolled_loop<0, 2 * N, 2>([&](int idx_out) {
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
 * @tparam T_ptr type of pointer for `in` and `out`. Can be raw pointer or sycl::multi_ptr.
 * @param in pointer to input
 * @param out pointer to output
 */
template <direction dir, int N, int M, int stride_in, int stride_out, typename T_ptr>
inline __attribute__((always_inline)) void cooley_tukey_dft(T_ptr in, T_ptr out) {
  using T = remove_ptr<T_ptr>;
  T tmp_buffer[2 * N * M];

  unrolled_loop<0, M, 1>([&](int i) __attribute__((always_inline)) {
    wi_dft<dir, N, M * stride_in, 1>(in + 2 * i * stride_in, tmp_buffer + 2 * i * N);
    unrolled_loop<0, N, 1>([&](int j) __attribute__((always_inline)) {
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
  unrolled_loop<0, N, 1>([&](int i) __attribute__((always_inline)) {
    wi_dft<dir, M, N, N * stride_out>(tmp_buffer + 2 * i, out + 2 * i * stride_out);
  });
}

/**
 * Factorizes a number into two roughly equal factors.
 * @param N the number to factorize
 * @return the smaller of the factors
 */
constexpr int factorize(int N) {
  int res = 1;
  for (int i = 2; i * i <= N; i++) {
    if (N % i == 0) {
      res = i;
    }
  }
  return res;
}

/**
 * Calculates how many temporary complex values a workitem implementation needs
 * for solving FFT.
 * @param N size of the FFT problem
 * @return Number of temporary complex values
 */
constexpr int wi_temps(int N) {
  int F0 = factorize(N);
  int F1 = N / F0;
  if (F0 < 2 || F1 < 2) {
    return N;
  }
  int a = wi_temps(F0);
  int b = wi_temps(F1);
  return (a > b ? a : b) + N;
}

/**
 * Checks whether a problem can be solved with workitem implementation without
 * registers spilling.
 * @tparam Scalar type of the real scalar used for the computation
 * @param N Size of the problem, in complex values
 * @return true if the problem fits in the registers
 */
template <typename Scalar>
constexpr bool fits_in_wi(int N) {
  int N_complex = N + wi_temps(N);
  int complex_size = 2 * sizeof(Scalar);
  int register_space = SYCLFFT_TARGET_REGS_PER_WI * 4;
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
 * @param N Size of the problem, in complex values
 * @return true if the problem fits in the registers
 */
template <typename Scalar>
bool fits_in_wi_device(int fft_size) {
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
 * @tparam T_ptr type of pointer for `in` and `out`. Can be raw pointer or sycl::multi_ptr.
 * @param in pointer to input
 * @param out pointer to output
 */
template <direction dir, int N, int stride_in, int stride_out, typename T_ptr>
inline __attribute__((always_inline)) void wi_dft(T_ptr in, T_ptr out) {
  constexpr int F0 = detail::factorize(N);
  if constexpr (N == 2) {
    using T = detail::remove_ptr<T_ptr>;
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
    detail::naive_dft<dir, N, stride_in, stride_out, T_ptr>(in, out);
  }
}

};  // namespace sycl_fft

#endif
