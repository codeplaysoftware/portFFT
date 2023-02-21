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
#include <common/helpers.hpp>
#include <common/twiddle.hpp>

namespace sycl_fft{
namespace detail{
    
//forward declaration
template <int N, int stride, typename T>
inline void cross_sg_dft(T& real, T& imag);

template<int N, int stride, typename T>
void __attribute__((always_inline)) cross_sg_naive_dft(T& real, T& imag, sycl::sub_group& sg){
    int local_id = sg.get_local_linear_id();
    int k = (local_id/stride) % N;
    int fft_start = local_id - k * stride;

    T res_real = 0;
    T res_imag = 0;
    
    unrolled_loop<0,N/stride,1>([&](int n) __attribute__((always_inline)) {
        const T multi_re = twiddle_re[N][n*k%N];
        const T multi_im = twiddle_im[N][n*k%N];

        int idx = fft_start + n * stride;

        T cur_real = sycl::select_from_group(sg, real, idx);
        T cur_imag = sycl::select_from_group(sg, imag, idx);

        //multiply cur and multi
        res_real += cur_real * multi_re - cur_imag * multi_im;
        res_imag += cur_real * multi_im + cur_imag * multi_re;
    });

    real = res_real;
    imag = res_imag;
}

template<int N, int M, int stride, typename T>
void cross_sg_transpose(T& real, T& imag, sycl::sub_group& sg){
    int local_id = sg.get_local_linear_id();
    int index_in_outer_dft = local_id/stride;
    int k = outer_dft_index % N; // in fft / 
    int n = outer_dft_index / N; // fft number / 
    int fft_start = local_id - k * stride;
    int target_index = fft_start + stride * (k * N + n);
    real = sycl::select_from_group(sg, real, target_index);
    imag = sycl::select_from_group(sg, imag, target_index);
}

template<int N, int M, int stride, typename T>
void __attribute__((always_inline)) cross_sg_cooley_tukey_dft(T& real, T& imag, sycl::sub_group& sg){
    constexpr T TWOPI = 2.0 * M_PI;

    int local_id = sg.get_local_linear_id();
    int k = local_id % N;
    int fft_start = local_id - k;
    T res_real = 0;
    T res_imag = 0;

    // factor N
    cross_sg_dft<N, M*stride>(real, imag);
    // transpose
    cross_sg_transpose<N,M, stride>(real, imag);
    // twiddle
    T tmp_real = real * twiddle_re[N*M][i*j] - imag * twiddle_im[N*M][i*j];
    imag = real * twiddle_im[N*M][i*j] + imag * twiddle_re[N*M][i*j];
    real = tmp_real;
    // factor M
    cross_sg_dft<M, N*stride>(real, imag);

    
    /*unrolled_loop<0,N,1>([&](int n) __attribute__((always_inline)) {
        const T multi_re = twiddle_re[N][n*k%N];
        const T multi_im = twiddle_im[N][n*k%N];

        int idx = fft_start + n * stride;

        T cur_real = sycl::select_from_group(sg, real, idx);
        T cur_imag = sycl::select_from_group(sg, imag, idx);

        //multiply cur and multi
        res_real += cur_real * multi_re - cur_imag * multi_im;
        res_imag += cur_real * multi_im + cur_imag * multi_re;
    });

    real = res_real;
    imag = res_imag;*/
}


template <int N, int stride, typename T>
inline void cross_sg_dft(T& real, T& imag){
    constexpr int F0 = detail::factorize<N>::factor;
    if constexpr(F0 >= 2 && N/F0 >= 2){
        cross_sg_cooley_tukey_dft<N/F0, F0, stride_in, stride_out>(in, out);
    } else {
        cross_sg_naive_dft<N, stride_in, stride_out>(in, out);
    }

}

void sg_dft(){

}

};
};

#endif