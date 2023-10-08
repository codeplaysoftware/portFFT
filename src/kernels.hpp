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

#ifndef PORTFFT_KERNELS_HPP
#define PORTFFT_KERNELS_HPP

#include <enums.hpp>

namespace portfft {
namespace detail {

// kernel names
// TODO: Remove all templates except Scalar, Domain and Memory and SubgroupSize
template <typename Scalar, domain, direction, detail::memory, detail::layout, detail::layout,
          detail::elementwise_multiply, detail::elementwise_multiply, detail::apply_scale_factor, Idx SubgroupSize>
class workitem_kernel;
template <typename Scalar, domain, direction, detail::memory, detail::layout, detail::layout,
          detail::elementwise_multiply, detail::elementwise_multiply, detail::apply_scale_factor, Idx SubgroupSize>
class subgroup_kernel;
template <typename Scalar, domain, direction, detail::memory, detail::layout, detail::layout,
          detail::elementwise_multiply, detail::elementwise_multiply, detail::apply_scale_factor, Idx SubgroupSize>
class workgroup_kernel;
template <typename Scalar, domain, direction, detail::memory, detail::layout, detail::layout,
          detail::elementwise_multiply, detail::elementwise_multiply, detail::apply_scale_factor, Idx SubgroupSize>
class global_kernel;

template <typename Scalar, domain Domain, detail::memory Mem, int SubgroupSize>
class transpose_kernel;

}  // namespace detail
}  // namespace portfft

#endif