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

#ifndef PORTFFT_SPECIALIZATION_CONSTANT_HPP
#define PORTFFT_SPECIALIZATION_CONSTANT_HPP

#include <sycl/sycl.hpp>

#include "defines.hpp"
#include "enums.hpp"

namespace portfft {
namespace detail {

#ifndef PORTFFT_USE_ADAPTIVECPP

constexpr static sycl::specialization_id<Idx> SpecConstFftSize{};
constexpr static sycl::specialization_id<Idx> SpecConstNumRealsPerFFT{};
constexpr static sycl::specialization_id<Idx> SpecConstWIScratchSize{};
constexpr static sycl::specialization_id<complex_storage> SpecConstComplexStorage{};
constexpr static sycl::specialization_id<detail::elementwise_multiply> SpecConstMultiplyOnLoad{};
constexpr static sycl::specialization_id<detail::elementwise_multiply> SpecConstMultiplyOnStore{};
constexpr static sycl::specialization_id<detail::apply_scale_factor> SpecConstApplyScaleFactor{};

constexpr static sycl::specialization_id<Idx> SubgroupFactorWISpecConst{};
constexpr static sycl::specialization_id<Idx> SubgroupFactorSGSpecConst{};

constexpr static sycl::specialization_id<level> GlobalSubImplSpecConst{};
constexpr static sycl::specialization_id<Idx> GlobalSpecConstLevelNum{};
constexpr static sycl::specialization_id<Idx> GlobalSpecConstNumFactors{};

#endif // PORTFFT_USE_ADAPTIVECPP

}  // namespace detail
}  // namespace portfft
#endif
