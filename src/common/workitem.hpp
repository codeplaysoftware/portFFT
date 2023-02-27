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

#include <sycl/sycl.hpp>
#include <common/helpers.hpp>
#include <common/twiddle.hpp>

namespace sycl_fft{

//forward declaration
template <int N, int stride_in, int stride_out, typename T_ptr>
inline void wi_dft(T_ptr in, T_ptr out);

namespace detail{

/**
 * Calculates DFT using naive algorithm. Can work in or out of place.
 * 
 * @tparam N size of the DFT transform
 * @tparam stride_in stride (in complex values) between complex values in `in`
 * @tparam stride_out stride (in complex values) between complex values in `out`
 * @tparam T_ptr type of pointer for `in` and `out`. Can be raw pointer or sycl::multi_ptr.
 * @param in pointer to input
 * @param out pointer to output
*/
template <int N, int stride_in, int stride_out, typename T_ptr>
inline __attribute__((always_inline)) void naive_dft(T_ptr in, T_ptr out) {
    using T = remove_ptr<T_ptr>;
    constexpr T TWOPI = 2.0 * M_PI;
    T tmp[2*N];
    unrolled_loop<0,N,1>([&](int k) __attribute__((always_inline)) {
        tmp[2*k+0] = 0;
        tmp[2*k+1] = 0;
        unrolled_loop<0,N,1>([&](int n) __attribute__((always_inline)) {
            // this multiplier is not really a twiddle factor, but it is calculated the same way
            const T multi_re = twiddle<T>::re[N][n*k%N];
            const T multi_im = twiddle<T>::im[N][n*k%N];

            // multiply in and multi
            tmp[2*k+0] += in[2*n*stride_in] * multi_re - in[2*n*stride_in + 1] * multi_im;
            tmp[2*k+1] += in[2*n*stride_in] * multi_im + in[2*n*stride_in + 1] * multi_re;
        });
    });
    unrolled_loop<0,2*N,2>([&](int k){
        out[k*stride_out+0] = tmp[k+0];
        out[k*stride_out+1] = tmp[k+1];
    });
}

//mem requirement: ~N*M(if in place, otherwise x2) + N*M(=tmp) + sqrt(N*M) + pow(N*M,0.25) + ...
// TODO explore if this tmp can be reduced/eliminated ^^^^^^
/**
 * Calculates DFT using Cooley-Tukey FFT algorithm. Can work in or out of place. Size of the problem is N*M
 * 
 * @tparam N the first factor of the problem size
 * @tparam M the second factor of the problem size
 * @tparam stride_in stride (in complex values) between complex values in `in`
 * @tparam stride_out stride (in complex values) between complex values in `out`
 * @tparam T_ptr type of pointer for `in` and `out`. Can be raw pointer or sycl::multi_ptr.
 * @param in pointer to input
 * @param out pointer to output
*/
template <int N, int M, int stride_in, int stride_out, typename T_ptr>
inline __attribute__((always_inline)) void cooley_tukey_dft(T_ptr in, T_ptr out) {
    using T = remove_ptr<T_ptr>;
    T tmp_buffer[2*N*M];
    unrolled_loop<0,M,1>([&](int i) __attribute__((always_inline)) {
        wi_dft<N, M*stride_in, 1>(in + 2*i*stride_in, tmp_buffer + 2*i*N);
        //wi_dft<N, M*stride_in, M*stride_out>(in + 2*i*stride_in, out + 2*i*stride_out);
        unrolled_loop<0,N,1>([&](int j) __attribute__((always_inline)) {
            T tmp_val = tmp_buffer[2*i*N + 2*j] * twiddle<T>::re[N*M][i*j] - tmp_buffer[2*i*N + 2*j + 1] * twiddle<T>::im[N*M][i*j];
            tmp_buffer[2*i*N + 2*j + 1] = tmp_buffer[2*i*N + 2*j] * twiddle<T>::im[N*M][i*j] + tmp_buffer[2*i*N + 2*j + 1] * twiddle<T>::re[N*M][i*j];
            tmp_buffer[2*i*N + 2*j + 0] = tmp_val;
        });
    });
    unrolled_loop<0,N,1>([&](int i) __attribute__((always_inline)) {
        wi_dft<M, N, N*stride_out>(tmp_buffer + 2*i, out + 2*i*stride_out);
    });
    /*unrolled_loop<0,N*M,1>([&](int i) __attribute__((always_inline)) {
      out[2*i*stride_out] = tmp_buffer[2*i];
      out[2*i*stride_out+1] = tmp_buffer[2*i+1];
    });*/
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

}; //namespace detail

/**
 * Calculates DFT using FFT algorithm. Can work in or out of place.
 * 
 * @tparam N size of the DFT transform
 * @tparam stride_in stride (in complex values) between complex values in `in`
 * @tparam stride_out stride (in complex values) between complex values in `out`
 * @tparam T_ptr type of pointer for `in` and `out`. Can be raw pointer or sycl::multi_ptr.
 * @param in pointer to input
 * @param out pointer to output
*/
template <int N, int stride_in, int stride_out, typename T_ptr>
inline __attribute__((always_inline)) void wi_dft(T_ptr in, T_ptr out){
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
    } else if constexpr(F0 >= 2 && N/F0 >= 2){
        detail::cooley_tukey_dft<N/F0, F0, stride_in, stride_out>(in, out);
    } else {
        detail::naive_dft<N, stride_in, stride_out>(in, out);
    }
}

}; //namespace sycl_fft

#endif
