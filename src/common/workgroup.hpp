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


#ifndef SYCL_FFT_COMMON_WORKGROUP_HPP
#define SYCL_FFT_COMMON_WORKGROUP_HPP

#include <common/subgroup.hpp>
#include <common/helpers.hpp>
#include <enums.hpp>

namespace sycl_fft {
//TODO: refactor code here
template <direction dir, int factor_sg, int factor_wi, typename T_ptr, typename T_twiddles_ptr>
__attribute__((always_inline)) inline void wg_dft(T_ptr inout, sycl::sub_group& sg) {
    
}

}

#endif