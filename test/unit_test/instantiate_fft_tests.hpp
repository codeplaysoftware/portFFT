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

constexpr test_placement_layouts_params valid_placement_layouts[] = {
    {placement::IN_PLACE, detail::layout::PACKED, detail::layout::PACKED},
    {placement::IN_PLACE, detail::layout::BATCH_INTERLEAVED, detail::layout::BATCH_INTERLEAVED},
    {placement::OUT_OF_PLACE, detail::layout::PACKED, detail::layout::PACKED},
    {placement::OUT_OF_PLACE, detail::layout::PACKED, detail::layout::BATCH_INTERLEAVED},
    {placement::OUT_OF_PLACE, detail::layout::BATCH_INTERLEAVED, detail::layout::BATCH_INTERLEAVED},
    {placement::OUT_OF_PLACE, detail::layout::BATCH_INTERLEAVED, detail::layout::PACKED}};

auto all_valid_placement_layouts = ::testing::ValuesIn(valid_placement_layouts);
auto fwd_only = ::testing::Values(direction::FORWARD);
auto bwd_only = ::testing::Values(direction::BACKWARD);

// sizes that use workitem implementation
INSTANTIATE_TEST_SUITE_P(workItemTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(
                             ::testing::Combine(all_valid_placement_layouts, fwd_only, ::testing::Values(1, 3, 33000),
                                                ::testing::Values(1, 2, 3, 4, 8))),
                         test_params_print());
// sizes that might use workitem or subgroup implementation depending on device
// and configurations
INSTANTIATE_TEST_SUITE_P(workItemOrSubgroupTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(all_valid_placement_layouts,
                                                                                           fwd_only,
                                                                                           ::testing::Values(1, 3, 555),
                                                                                           ::testing::Values(16, 32))),
                         test_params_print());
// sizes that use subgroup implementation
INSTANTIATE_TEST_SUITE_P(SubgroupTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(
                             ::testing::Combine(all_valid_placement_layouts, fwd_only, ::testing::Values(1, 3, 555),
                                                ::testing::Values(64, 96, 128))),
                         test_params_print());

INSTANTIATE_TEST_SUITE_P(SubgroupOrWorkgroupTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(
                             ::testing::Combine(all_valid_placement_layouts, fwd_only, ::testing::Values(1, 3),
                                                ::testing::Values(256, 512, 1024))),
                         test_params_print());

INSTANTIATE_TEST_SUITE_P(WorkgroupTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(
                             ::testing::Combine(all_valid_placement_layouts, fwd_only, ::testing::Values(1, 3),
                                                ::testing::Values(2048, 3072, 4096))),
                         test_params_print());

// Backward FFT test suite
INSTANTIATE_TEST_SUITE_P(BackwardTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(
                             ::testing::Combine(all_valid_placement_layouts, bwd_only, ::testing::Values(1, 3),
                                                ::testing::Values(8, 9, 16, 32, 64, 4096))),
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

#define INSTANTIATE_TESTS(TYPE)     \
  INSTANTIATE_TESTS_FULL(TYPE, usm) \
  INSTANTIATE_TESTS_FULL(TYPE, buffer)

#endif
