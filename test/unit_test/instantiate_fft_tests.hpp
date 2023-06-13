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

#ifndef SYCL_FFT_UNIT_TEST_INSTANTIATE_FFT_TESTS_HPP
#define SYCL_FFT_UNIT_TEST_INSTANTIATE_FFT_TESTS_HPP

#include "fft_test_utils.hpp"
#include <gtest/gtest.h>

class FFTTest : public ::testing::TestWithParam<test_params> {};  // batch, length
class BwdTest : public ::testing::TestWithParam<test_params> {};  // batch, length

// sizes that use workitem implementation
INSTANTIATE_TEST_SUITE_P(workItemTest, FFTTest,
                         ::testing::ConvertGenerator<param_tuple>(
                             ::testing::Combine(::testing::Values(1, 3, 33000), ::testing::Values(1, 2, 4, 8, 16))));
// sizes that might use workitem or subgroup implementation depending on device
// and configuration
INSTANTIATE_TEST_SUITE_P(workItemOrSubgroupTest, FFTTest,
                         ::testing::ConvertGenerator<param_tuple>(::testing::Combine(::testing::Values(1, 3, 555),
                                                                                     ::testing::Values(32))));
// sizes that use subgroup implementation
INSTANTIATE_TEST_SUITE_P(SubgroupTest, FFTTest,
                         ::testing::ConvertGenerator<param_tuple>(::testing::Combine(::testing::Values(1, 3, 555),
                                                                                     ::testing::Values(64))));
// Backward FFT test suite
INSTANTIATE_TEST_SUITE_P(BackwardFFT, BwdTest,
                         ::testing::ConvertGenerator<param_tuple>(
                             ::testing::Combine(::testing::Values(1), ::testing::Values(8, 16, 32, 64))));

#define INTANTIATE_TESTS(TYPE, TYPE_NAME, PLACEMENT, PLACEMENT_NAME, TRANSPOSE, TRANSPOSE_NAME, DIRECTION,         \
                         DIRECTION_NAME, DIRECTION_TEST_SUITE, MEM, MEM_NAME)                                      \
  TEST_P(DIRECTION_TEST_SUITE, MEM_NAME##_##PLACEMENT_NAME##_C2C_##DIRECTION_NAME##_##TYPE_NAME##TRANSPOSE_NAME) { \
    auto param = GetParam();                                                                                       \
    sycl::queue queue;                                                                                             \
    check_fft_##MEM<TYPE, placement::PLACEMENT, direction::DIRECTION, TRANSPOSE>(param, queue);                    \
  }

#define INTANTIATE_TESTS_MEM(TYPE, TYPE_NAME, PLACEMENT, PLACEMENT_NAME, TRANSPOSE, TRANSPOSE_NAME, DIRECTION,       \
                             DIRECTION_NAME, DIRECTION_TEST_SUITE)                                                   \
  INTANTIATE_TESTS(TYPE, TYPE_NAME, PLACEMENT, PLACEMENT_NAME, TRANSPOSE, TRANSPOSE_NAME, DIRECTION, DIRECTION_NAME, \
                   DIRECTION_TEST_SUITE, usm, USM)                                                                   \
  INTANTIATE_TESTS(TYPE, TYPE_NAME, PLACEMENT, PLACEMENT_NAME, TRANSPOSE, TRANSPOSE_NAME, DIRECTION, DIRECTION_NAME, \
                   DIRECTION_TEST_SUITE, buffer, BUFFER)

#define INTANTIATE_TESTS_MEM_DIRECTION(TYPE, TYPE_NAME, PLACEMENT, PLACEMENT_NAME, TRANSPOSE, TRANSPOSE_NAME)        \
  INTANTIATE_TESTS_MEM(TYPE, TYPE_NAME, PLACEMENT, PLACEMENT_NAME, TRANSPOSE, TRANSPOSE_NAME, FORWARD, Fwd, FFTTest) \
  INTANTIATE_TESTS_MEM(TYPE, TYPE_NAME, PLACEMENT, PLACEMENT_NAME, TRANSPOSE, TRANSPOSE_NAME, BACKWARD, Bwd, BwdTest)

#define INTANTIATE_TESTS_MEM_DIRECTION_PLACEMENT_TRANSPOSE(TYPE, TYPE_NAME)   \
  INTANTIATE_TESTS_MEM_DIRECTION(TYPE, TYPE_NAME, IN_PLACE, IP, false, )      \
  INTANTIATE_TESTS_MEM_DIRECTION(TYPE, TYPE_NAME, OUT_OF_PLACE, OOP, false, ) \
  INTANTIATE_TESTS_MEM_DIRECTION(TYPE, TYPE_NAME, OUT_OF_PLACE, OOP, true, _in_transposed)
// transpose in place is not supported (yet?)

#endif
