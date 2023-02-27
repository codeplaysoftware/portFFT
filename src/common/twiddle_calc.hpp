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

#ifndef SYCL_FFT_COMMON_TWIDDLE_CALC_HPP
#define SYCL_FFT_COMMON_TWIDDLE_CALC_HPP

#include <complex>
#include <cmath>

namespace sycl_fft{
namespace detail{

/**
 * Calculates a twiddle factor.
 * @tparam T floating point type to use
 * @param n which twiddle factor to calculate
 * @param total total number of twiddles
*/
template<typename T>
std::complex<T> calculate_twiddle(std::size_t n, std::size_t total){
    T theta = -2 * M_PI * n / total;
    return {std::cos(theta), std::sin(theta)};
}

}
}

#endif