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
#include <sycl/sycl.hpp>
#include <type_traits>

namespace sycl_fft::detail{



template<typename T>
struct remove_multi_ptr{
    using type = T;
};

#ifdef SYCL_IMPLEMENTATION_ONEAPI
// OneAPI Adds decorate address space during version 6.1, so this will
// not work in all cases.
#if __LIBSYCL_MAJOR_VERSION >= 6 & __LIBSYCL_MINOR_VERSION >= 1
#define SYCLFFT_SYCL_HAS_DECORATEADDRSPACE
#endif // __LIBSYCL version
#endif // SYCL_IMPLEMENTATION_ONEAPI
#ifdef SYCLFFT_SYCL_HAS_DECORATEADDRSPACE
template<typename T, sycl::access::address_space Space,
          sycl::access::decorated DecorateAddress>
struct remove_multi_ptr<sycl::multi_ptr<T, Space, DecorateAddress>>{
    using type = T;
};
#else
template<typename T, sycl::access::address_space Space>
struct remove_multi_ptr<sycl::multi_ptr<T, Space>>{
    using type = T;
};
#endif // SYCLFFT_SYCL_HAS_DECORATEADDRSPACE
#undef SYCLFFT_SYCL_HAS_DECORATEADDRSPACE

/**
 * Removes pointer or sycl::multi_ptr.
 * @tparam T type to remove pointer from
*/
template<typename T>
using remove_ptr = std::conditional_t<std::is_pointer_v<T>, std::remove_pointer_t<T>, typename remove_multi_ptr<T>::type>;

/**
 * Implements a loop that will be fully unrolled.
 * @tparam Start starting value of loop counter
 * @tparam Stop loop counter value before which the loop terminates
 * @tparam Step Increment of the loop counter
 * @tparam Functor type of the callable
 * @param funct functor containing body of the loop. Should accept one value - the loop counter. Should have __attribute__((always_inline)).
*/
template<int Start, int Stop, int Step, typename Functor>
void __attribute__((always_inline)) unrolled_loop(Functor&& funct){
    if constexpr (Start<Stop){
        funct(Start);
        unrolled_loop<Start+Step, Stop, Step>(funct);
    }
}

};