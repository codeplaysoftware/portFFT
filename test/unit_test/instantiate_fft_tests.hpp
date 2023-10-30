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

#ifndef PORTFFT_UNIT_TEST_INSTANTIATE_FFT_TESTS_HPP
#define PORTFFT_UNIT_TEST_INSTANTIATE_FFT_TESTS_HPP

#include <gtest/gtest.h>
#include <type_traits>

#include "fft_test_utils.hpp"
#include <common/exceptions.hpp>

// Parameters: placement, layout, direction, batch, length
class FFTTest : public ::testing::TestWithParam<test_params> {};

using sizes_t = std::vector<std::size_t>;

constexpr test_placement_layouts_params valid_placement_layouts[] = {
    {placement::IN_PLACE, detail::layout::PACKED, detail::layout::PACKED}};
auto all_valid_placement_layouts = ::testing::ValuesIn(valid_placement_layouts);

constexpr test_placement_layouts_params valid_multi_dim_placement_layouts[] = {
    {placement::IN_PLACE, detail::layout::PACKED, detail::layout::PACKED}};
auto all_valid_multi_dim_placement_layouts = ::testing::ValuesIn(valid_multi_dim_placement_layouts);

auto fwd_only = ::testing::Values(direction::FORWARD);
auto bwd_only = ::testing::Values(direction::BACKWARD);
auto both_directions = ::testing::Values(direction::FORWARD, direction::BACKWARD);

// sizes that use workgroup implementation
INSTANTIATE_TEST_SUITE_P(GlobalTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_placement_layouts, fwd_only, ::testing::Values(1),
                             ::testing::Values(sizes_t{32768} /*, sizes_t{16384}, sizes_t{32768}, sizes_t{65536}*/))),
                         test_params_print());

#define INSTANTIATE_TESTS_FULL(TYPE, MEMORY)                                     \
  TEST_P(FFTTest, TYPE##_##MEMORY##_C2C) {                                       \
    auto params = GetParam();                                                    \
    if (params.dir == portfft::direction::FORWARD) {                             \
      run_test<test_memory::MEMORY, TYPE, portfft::direction::FORWARD>(params);  \
    } else {                                                                     \
      run_test<test_memory::MEMORY, TYPE, portfft::direction::BACKWARD>(params); \
    }                                                                            \
  }

#define INSTANTIATE_TESTS(TYPE) INSTANTIATE_TESTS_FULL(TYPE, usm)

#endif
