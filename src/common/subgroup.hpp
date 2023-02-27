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

#ifndef SYCL_FFT_COMMON_SOUBGROUP_HPP
#define SYCL_FFT_COMMON_SOUBGROUP_HPP

#include <sycl/sycl.hpp>
#include <common/twiddle_calc.hpp>
#include <common/helpers.hpp>
#include <common/twiddle.hpp>
#include <common/workitem.hpp>

namespace sycl_fft{
namespace detail{
    
//forward declaration
template <int N, int stride, typename T>
inline void cross_sg_dft(T& real, T& imag, sycl::sub_group& sg);

template<int N, int stride, typename T>
void __attribute__((always_inline)) cross_sg_naive_dft(T& real, T& imag, sycl::sub_group& sg){
    int local_id = sg.get_local_linear_id();
    int k = (local_id/stride) % N;
    int fft_start = local_id - k * stride;

    T res_real = 0;
    T res_imag = 0;
    
    unrolled_loop<0,N,1>([&](int n) __attribute__((always_inline)) {
        const T multi_re = twiddle<T>::re[N][n*k%N];
        const T multi_im = twiddle<T>::im[N][n*k%N];

        int idx = fft_start + n * stride;

        T cur_real = sycl::group_broadcast(sg, real, idx);
        T cur_imag = sycl::group_broadcast(sg, imag, idx);

        //multiply cur and multi
        res_real += cur_real * multi_re - cur_imag * multi_im;
        res_imag += cur_real * multi_im + cur_imag * multi_re;
    });

    real = res_real;
    imag = res_imag;
}

template<int N, int M, int stride, typename T>
void __attribute__((always_inline)) cross_sg_transpose(T& real, T& imag, sycl::sub_group& sg){
    int local_id = sg.get_local_linear_id();
    int index_in_outer_dft = (local_id/stride) % (N * M);
    int k = index_in_outer_dft % N; // in fft
    int n = index_in_outer_dft / N; // fft number
    int fft_start = local_id - index_in_outer_dft * stride;
    int target_index = fft_start + stride * (k * M + n);
    real = sycl::select_from_group(sg, real, target_index);
    imag = sycl::select_from_group(sg, imag, target_index);
}

template<int N, int M, int stride, typename T>
void __attribute__((always_inline)) cross_sg_cooley_tukey_dft(T& real, T& imag, sycl::sub_group& sg){
    int local_id = sg.get_local_linear_id();
    int index_in_outer_dft = (local_id/stride) % (N * M);
    int k = index_in_outer_dft % N; // in fft
    int n = index_in_outer_dft / N; // fft number

    // factor N
    cross_sg_dft<N, M*stride>(real, imag, sg);
    // transpose
    cross_sg_transpose<N, M, stride>(real, imag, sg);
    // twiddle
    T tmp_real = real * twiddle<T>::re[N*M][k*n] - imag * twiddle<T>::im[N*M][k*n];
    imag = real * twiddle<T>::im[N*M][k*n] + imag * twiddle<T>::re[N*M][k*n];
    real = tmp_real;
    // factor M
    cross_sg_dft<M, N*stride>(real, imag, sg);
}


template <int N, int stride, typename T>
inline void __attribute__((always_inline)) cross_sg_dft(T& real, T& imag, sycl::sub_group& sg){
    constexpr int F0 = detail::factorize(N);
    if constexpr(F0 >= 2 && N/F0 >= 2){
        cross_sg_cooley_tukey_dft<N/F0, F0, stride>(real, imag, sg);
    } else {
        cross_sg_naive_dft<N, stride>(real, imag, sg);
    }
}

/**
 * Factorizes a number into two factors, so that one of them will maximal below or equal to subgroup size.
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


template<typename T_ptr>
void subgroup_to_workitem_dispatcher(int fft_size, T_ptr in, T_ptr out){
    using T = remove_ptr<T_ptr>;
    switch(fft_size){
#define SYCL_FFT_SG_WI_DISPATCHER_IMPL(N)                                   \
  case N:                                                                \
    if constexpr (fits_in_wi<T>(N)) {                                    \
      wi_dft<N,1,1>(in, out);                                          \
    }                                                                    \
    break;
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(1)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(2)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(3)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(4)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(5)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(6)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(7)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(8)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(9)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(10)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(11)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(12)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(13)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(14)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(15)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(16)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(17)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(18)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(19)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(20)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(21)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(22)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(23)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(24)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(25)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(26)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(27)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(28)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(29)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(30)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(31)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(32)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(33)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(34)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(35)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(36)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(37)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(38)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(39)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(40)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(41)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(42)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(43)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(44)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(45)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(46)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(47)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(48)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(49)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(50)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(51)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(52)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(53)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(54)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(55)
      SYCL_FFT_SG_WI_DISPATCHER_IMPL(56)
#undef SYCL_FFT_SG_WI_DISPATCHER_IMPL
}

};

};
/**
 * Calculates FFT of size N*M using workitems in a subgroup. Works in place. The end result needs to be transposed when storing it to the local memory!
 * @tparam N number of workitems in a subgroup that work on one FFT
 * @tparam M number of elements per workitem
 * @tparam T_prt type of the pointer to the data
 * @param inout pointer to private memory where the input/output data is
 * @param sg subgroup
 * @param sg_twiddles twiddle factors to use - calculated by sg_calc_twiddles in commit
*/
template<int N, int M, typename T_ptr>
void sg_dft(T_ptr inout, sycl::sub_group& sg, const T_ptr sg_twiddles, sycl::stream s){
    using T = detail::remove_ptr<T_ptr>;
    int n = sg.get_local_linear_id();


    detail::unrolled_loop<0,M,1>([&](int k) __attribute__((always_inline)) {
        T& real = inout[2*k];
        T& imag = inout[2*k+1];

        detail::cross_sg_dft<N, 1>(real, imag, sg);

        //T tmp_real = real * detail::twiddle<T>::re[N*M][k*n] - imag * detail::twiddle<T>::im[N*M][k*n];
        //imag = real * detail::twiddle<T>::im[N*M][k*n] + imag * detail::twiddle<T>::re[N*M][k*n];
        T twiddle_real = sg_twiddles[k * N + n];
        T twiddle_imag = sg_twiddles[(k + M) * N + n];
        std::complex<T> twiddle = detail::calculate_twiddle<T>(n*k, N*M);
        //s << "twiddle in" << real << " " << imag << " nl\n";
        T tmp_real = real * twiddle_real - imag * twiddle_imag;
        imag = real * twiddle_imag + imag * twiddle_real;
        real = tmp_real;
        s << "twiddle res " << real << " " << imag << " nl\n";
    });

    wi_dft<M,1,1>(inout, inout);
}

template<typename T_ptr>
void sg_calc_twiddles(int N, int M, int n, int k, T_ptr sg_twiddles){
  using T = detail::remove_ptr<T_ptr>;
  std::complex<T> twiddle = detail::calculate_twiddle<T>(n*k, N*M);
  sg_twiddles[k * N + n] = twiddle.real();
  sg_twiddles[(k + M) * N + n] = twiddle.imag();
}

};

#endif