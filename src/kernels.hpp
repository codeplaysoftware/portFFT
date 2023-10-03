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
template <typename Scalar, domain Domain, direction Dir, detail::memory, detail::layout LayoutIn,
          detail::layout LayoutOut, apply_load_modifier ApplyLoadModifier, apply_store_modifier ApplyStoreModifier,
          apply_scale_factor ApplyScaleFactor, int SubgroupSize>
class workitem_kernel;
template <typename Scalar, domain Domain, direction Dir, detail::memory, detail::layout LayoutIn,
          detail::layout LayoutOut, apply_load_modifier ApplyLoadModifier, apply_store_modifier ApplyStoreModifier,
          apply_scale_factor ApplyScaleFactor, int SubgroupSize>
class subgroup_kernel;
template <typename Scalar, domain Domain, direction Dir, detail::memory, detail::layout LayoutIn,
          detail::layout LayoutOut, apply_load_modifier ApplyLoadModifier, apply_store_modifier ApplyStoreModifier,
          apply_scale_factor ApplyScaleFactor, int SubgroupSize>
class workgroup_kernel;
template <typename Scalar, domain Domain, direction Dir, detail::memory, detail::layout LayoutIn,
          detail::layout LayoutOut, apply_load_modifier ApplyLoadModifier, apply_store_modifier ApplyStoreModifier,
          apply_scale_factor ApplyScaleFactor, int SubgroupSize>
class global_kernel;

template <typename Scalar, domain Domain, detail::memory Mem, int SubgroupSize>
class transpose_kernel;

}  // namespace detail
}  // namespace portfft

#endif