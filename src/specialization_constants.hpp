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

#ifndef PORTFFT_SPECIALIZATION_CONSTANTS_HPP
#define PORTFFT_SPECIALIZATION_CONSTANTS_HPP

#include <sycl/sycl.hpp>

namespace portfft {
namespace detail {
constexpr static sycl::specialization_id<std::size_t> WorkitemSpecConstFftSize{};
constexpr static sycl::specialization_id<int> FactorWISpecConst{};
constexpr static sycl::specialization_id<int> FactorSGSpecConst{};
constexpr static sycl::specialization_id<std::size_t> WorkgroupSpecConstFftSize{};
constexpr static sycl::specialization_id<std::size_t> GlobalSpecConstFftSize{};
constexpr static sycl::specialization_id<int> GlobalSpecConstSGFactorWI{};
constexpr static sycl::specialization_id<int> GlobalSpecConstSGFactorSG{};
constexpr static sycl::specialization_id<level> GlobalSpecConstLevel{};
constexpr static sycl::specialization_id<std::size_t> GlobalSpecConstNumFactors{};
constexpr static sycl::specialization_id<std::size_t> GlobalSpecConstLevelNum{};

}  // namespace detail
}  // namespace portfft

#endif