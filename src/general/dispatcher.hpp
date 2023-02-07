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

#ifndef SYCL_FFT_GENERAL_DISPATCHER_HPP
#define SYCL_FFT_GENERAL_DISPATCHER_HPP

#include <common/helpers.hpp>
#include <common/workitem.hpp>
#include <common/transfers.hpp>

namespace sycl_fft{

namespace detail{

/**
 * Implementation of FFT for sizes that can be done by independent work items.
 * 
 * @tparam N size of each transform
 * @tparam T_in type of the accessor or pointer to global memory containing input data
 * @tparam T_out type of the accessor or pointer to global memory for output data
 * @tparam T type of the scalar used for computations
 * @param input accessor or pointer to global memory containing input data
 * @param output accessor or pointer to global memory for output data
 * @param loc local accessor. Must have enough space for 2*N*subgroup_size values
 * @param n_transforms number of FT transforms to do in one call
 * @param input_distance Distance between data for two FFT transforms within input data
 * @param output_distance Distance between data for two FFT transforms within output data
 * @param it sycl::nd_item<1> for the kernel launch
 */
template<int N, typename T_in, typename T_out, typename T>
inline void workitem_impl(T_in input, T_out output, const sycl::local_accessor<T,1>& loc, std::size_t n_transforms, 
                   std::size_t input_distance, std::size_t output_distance, sycl::nd_item<1> it){
    constexpr int N_reals = 2*N;

    T priv[N_reals];
    sycl::sub_group sg = it.get_sub_group();
    std::size_t subgroup_id = sg.get_local_linear_id();
    std::size_t global_id = it.get_global_id(0);
    std::size_t subgroup_size = sg.get_local_linear_range();
    std::size_t global_size = it.get_global_range(0);

    for(size_t i = global_id; i < roundUpToMultiple(n_transforms, subgroup_size); i+=global_size){
        bool working = i < n_transforms;
        int n_working = sycl::min(subgroup_size, n_transforms - i + global_id);

        global2local(input + input_distance * (i - global_id), loc, N_reals*n_working, subgroup_size, subgroup_id);
        sycl::group_barrier(sg);
        if(working){
            local2private<N_reals>(loc, priv, subgroup_id, N_reals);
            wi_dft<N,1,1>(priv, priv);
            private2local<N_reals>(priv, loc, subgroup_id, N_reals);
        }
        sycl::group_barrier(sg);
        local2global(loc, output + output_distance * (i - global_id), N_reals*n_working, subgroup_size, subgroup_id);
        sycl::group_barrier(sg);
    }
}

/**
 * Dispatcher to workitem implementations of FFT.
 * 
 * @tparam T_in type of the accessor or pointer to global memory containing input data
 * @tparam T_out type of the accessor or pointer to global memory for output data
 * @tparam T type of the scalar used for computations
 * @param input accessor or pointer to global memory containing input data
 * @param output accessor or pointer to global memory for output data
 * @param loc local accessor. Must have enough space for 2*N*subgroup_size values
 * @param fft_size size of each transform
 * @param n_transforms number of FT transforms to do in one call
 * @param input_distance Distance between data for two FFT transforms within input data
 * @param output_distance Distance between data for two FFT transforms within output data
 * @param it sycl::nd_item<1> for the kernel launch
 */
template<typename T_in, typename T_out, typename T>
void workitem_dispatcher(T_in input, T_out output, const sycl::local_accessor<T,1>& loc, std::size_t fft_size,
                   std::size_t n_transforms, std::size_t input_distance, std::size_t output_distance, sycl::nd_item<1> it){
    // TODO curretnly this does not work on intel GPU it was tested on.
    // The whole kernel might be too big. Commenting out implementations for sizes above 11 worked as a workaround.
    // Any tests must also be adjusted so that they do not test such sizes.
    switch(fft_size){
        #define SYCL_FFT_WI_DISPATCHER_IMPL(N) case N: workitem_impl<N>(input, output, loc, n_transforms, input_distance, output_distance, it); break;
        SYCL_FFT_WI_DISPATCHER_IMPL(1)
        SYCL_FFT_WI_DISPATCHER_IMPL(2)
        SYCL_FFT_WI_DISPATCHER_IMPL(3)
        SYCL_FFT_WI_DISPATCHER_IMPL(4)
        SYCL_FFT_WI_DISPATCHER_IMPL(5)
        SYCL_FFT_WI_DISPATCHER_IMPL(6)
        SYCL_FFT_WI_DISPATCHER_IMPL(7)
        SYCL_FFT_WI_DISPATCHER_IMPL(8)
        SYCL_FFT_WI_DISPATCHER_IMPL(9)
        SYCL_FFT_WI_DISPATCHER_IMPL(10)
        SYCL_FFT_WI_DISPATCHER_IMPL(11)
        SYCL_FFT_WI_DISPATCHER_IMPL(12)
        SYCL_FFT_WI_DISPATCHER_IMPL(13)
        SYCL_FFT_WI_DISPATCHER_IMPL(14)
        SYCL_FFT_WI_DISPATCHER_IMPL(15)
        SYCL_FFT_WI_DISPATCHER_IMPL(16)
        SYCL_FFT_WI_DISPATCHER_IMPL(17)
        SYCL_FFT_WI_DISPATCHER_IMPL(18)
        SYCL_FFT_WI_DISPATCHER_IMPL(19)
        SYCL_FFT_WI_DISPATCHER_IMPL(20)
        SYCL_FFT_WI_DISPATCHER_IMPL(21)
        SYCL_FFT_WI_DISPATCHER_IMPL(22)
        SYCL_FFT_WI_DISPATCHER_IMPL(23)
        SYCL_FFT_WI_DISPATCHER_IMPL(24)
        SYCL_FFT_WI_DISPATCHER_IMPL(25)
        SYCL_FFT_WI_DISPATCHER_IMPL(26)
        SYCL_FFT_WI_DISPATCHER_IMPL(27)
        SYCL_FFT_WI_DISPATCHER_IMPL(28)
        SYCL_FFT_WI_DISPATCHER_IMPL(29)
        SYCL_FFT_WI_DISPATCHER_IMPL(30)
        SYCL_FFT_WI_DISPATCHER_IMPL(31)
        SYCL_FFT_WI_DISPATCHER_IMPL(32)
        SYCL_FFT_WI_DISPATCHER_IMPL(33)
        SYCL_FFT_WI_DISPATCHER_IMPL(34)
        SYCL_FFT_WI_DISPATCHER_IMPL(35)
        SYCL_FFT_WI_DISPATCHER_IMPL(36)
        SYCL_FFT_WI_DISPATCHER_IMPL(37)
        SYCL_FFT_WI_DISPATCHER_IMPL(38)
        SYCL_FFT_WI_DISPATCHER_IMPL(39)
        SYCL_FFT_WI_DISPATCHER_IMPL(40)
        SYCL_FFT_WI_DISPATCHER_IMPL(41)
        SYCL_FFT_WI_DISPATCHER_IMPL(42)
        SYCL_FFT_WI_DISPATCHER_IMPL(43)
        SYCL_FFT_WI_DISPATCHER_IMPL(44)
        SYCL_FFT_WI_DISPATCHER_IMPL(45)
        SYCL_FFT_WI_DISPATCHER_IMPL(46)
        SYCL_FFT_WI_DISPATCHER_IMPL(47)
        SYCL_FFT_WI_DISPATCHER_IMPL(48)
        SYCL_FFT_WI_DISPATCHER_IMPL(49)
        SYCL_FFT_WI_DISPATCHER_IMPL(50)
        SYCL_FFT_WI_DISPATCHER_IMPL(51)
        SYCL_FFT_WI_DISPATCHER_IMPL(52)
        SYCL_FFT_WI_DISPATCHER_IMPL(53)
        SYCL_FFT_WI_DISPATCHER_IMPL(54)
        SYCL_FFT_WI_DISPATCHER_IMPL(55)
        SYCL_FFT_WI_DISPATCHER_IMPL(56)
        SYCL_FFT_WI_DISPATCHER_IMPL(57)
        SYCL_FFT_WI_DISPATCHER_IMPL(58)
        SYCL_FFT_WI_DISPATCHER_IMPL(59)
        SYCL_FFT_WI_DISPATCHER_IMPL(60)
        SYCL_FFT_WI_DISPATCHER_IMPL(61)
        SYCL_FFT_WI_DISPATCHER_IMPL(62)
        SYCL_FFT_WI_DISPATCHER_IMPL(63)
        SYCL_FFT_WI_DISPATCHER_IMPL(64)
        #undef SYCL_FFT_WI_DISPATCHER_IMPL
    }
}

}
}

#endif
