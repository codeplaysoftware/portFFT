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

class FFTTest : public ::testing::TestWithParam<test_params> {};  // batch, length
class BwdTest : public ::testing::TestWithParam<test_params> {};  // batch, length

// sizes that use workitem implementation
INSTANTIATE_TEST_SUITE_P(workItemTest, FFTTest,
                         ::testing::ConvertGenerator<param_tuple>(
                             ::testing::Combine(::testing::Values(1, 3, 33000), ::testing::Values(1, 2, 3, 4, 8))));
// sizes that might use workitem or subgroup implementation depending on device
// and configurations
INSTANTIATE_TEST_SUITE_P(workItemOrSubgroupTest, FFTTest,
                         ::testing::ConvertGenerator<param_tuple>(::testing::Combine(::testing::Values(1, 3, 555),
                                                                                     ::testing::Values(16, 32))));
// sizes that use subgroup implementation
INSTANTIATE_TEST_SUITE_P(SubgroupTest, FFTTest,
                         ::testing::ConvertGenerator<param_tuple>(::testing::Combine(::testing::Values(1, 3, 555),
                                                                                     ::testing::Values(64, 96, 128))));

INSTANTIATE_TEST_SUITE_P(SubgroupOrWorkgroupTest, FFTTest,
                         ::testing::ConvertGenerator<param_tuple>(
                             ::testing::Combine(::testing::Values(1, 3), ::testing::Values(256, 512, 1024))));

INSTANTIATE_TEST_SUITE_P(WorkgroupTest, FFTTest,
                         ::testing::ConvertGenerator<param_tuple>(
                             ::testing::Combine(::testing::Values(1, 3), ::testing::Values(2048, 3072, 4096))));

// Backward FFT test suite
INSTANTIATE_TEST_SUITE_P(BackwardFFT, BwdTest,
                         ::testing::ConvertGenerator<param_tuple>(
                             ::testing::Combine(::testing::Values(1, 3), ::testing::Values(8, 9, 16, 32, 64, 4096))));

#define INTANTIATE_TESTS(TYPE, TYPE_NAME, PLACEMENT, PLACEMENT_NAME, LAYOUT, LAYOUT_NAME, DIRECTION, DIRECTION_NAME, \
                         DIRECTION_TEST_SUITE, MEM, MEM_NAME)                                                        \
  TEST_P(DIRECTION_TEST_SUITE, MEM_NAME##_##PLACEMENT_NAME##_C2C_##DIRECTION_NAME##_##TYPE_NAME##LAYOUT_NAME) {      \
    auto param = GetParam();                                                                                         \
    sycl::queue queue;                                                                                               \
    if constexpr (std::is_same<TYPE, double>::value) {                                                               \
      auto queue_pair = get_queue(fp64_selector);                                                                    \
      CHECK_QUEUE(queue_pair);                                                                                       \
      queue = queue_pair.first.value();                                                                              \
    }                                                                                                                \
    check_fft_##MEM<TYPE, placement::PLACEMENT, direction::DIRECTION, LAYOUT>(param, queue);                         \
  }

#define INTANTIATE_TESTS_MEM(TYPE, TYPE_NAME, PLACEMENT, PLACEMENT_NAME, LAYOUT, LAYOUT_NAME, DIRECTION,       \
                             DIRECTION_NAME, DIRECTION_TEST_SUITE)                                             \
  INTANTIATE_TESTS(TYPE, TYPE_NAME, PLACEMENT, PLACEMENT_NAME, LAYOUT, LAYOUT_NAME, DIRECTION, DIRECTION_NAME, \
                   DIRECTION_TEST_SUITE, usm, USM)                                                             \
  INTANTIATE_TESTS(TYPE, TYPE_NAME, PLACEMENT, PLACEMENT_NAME, LAYOUT, LAYOUT_NAME, DIRECTION, DIRECTION_NAME, \
                   DIRECTION_TEST_SUITE, buffer, BUFFER)

#define INTANTIATE_TESTS_MEM_DIRECTION(TYPE, TYPE_NAME, PLACEMENT, PLACEMENT_NAME, LAYOUT, LAYOUT_NAME)        \
  INTANTIATE_TESTS_MEM(TYPE, TYPE_NAME, PLACEMENT, PLACEMENT_NAME, LAYOUT, LAYOUT_NAME, FORWARD, Fwd, FFTTest) \
  INTANTIATE_TESTS_MEM(TYPE, TYPE_NAME, PLACEMENT, PLACEMENT_NAME, LAYOUT, LAYOUT_NAME, BACKWARD, Bwd, BwdTest)

#define INTANTIATE_TESTS_MEM_DIRECTION_PLACEMENT_LAYOUT(TYPE, TYPE_NAME)                                \
  INTANTIATE_TESTS_MEM_DIRECTION(TYPE, TYPE_NAME, IN_PLACE, IP, detail::layout::PACKED, )               \
  INTANTIATE_TESTS_MEM_DIRECTION(TYPE, TYPE_NAME, OUT_OF_PLACE, OOP, detail::layout::PACKED, )          \
  INTANTIATE_TESTS_MEM_DIRECTION(TYPE, TYPE_NAME, OUT_OF_PLACE, OOP, detail::layout::BATCH_INTERLEAVED, \
                                 _in_batch_interleaved)
// BATCH_INTERLEAVED inplace is not supported (yet?)

#endif
