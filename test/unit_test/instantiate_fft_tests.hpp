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

#include <gtest/gtest.h>

using param_tuple = std::tuple<int, int>;

struct test_params {
  int batch;
  int length;
  test_params(param_tuple params) : batch(std::get<0>(params)), length(std::get<1>(params)) {}
};

void operator<<(std::ostream& stream, const test_params& params) {
  stream << "Batch = " << params.batch << ", Length = " << params.length << '\n';
}

class FFTTest : public ::testing::TestWithParam<test_params> {};  // batch, length
class BwdTest : public ::testing::TestWithParam<test_params> {};  // batch, length

// sizes that use workitem implementation
INSTANTIATE_TEST_SUITE_P(workItemTest, FFTTest,
                         ::testing::ConvertGenerator<param_tuple>(::testing::Combine(::testing::Values(1, 33, 32000),
                                                                                     ::testing::Range(1, 14))));
// sizes that might use workitem or subgroup implementation depending on device
// and configuration
INSTANTIATE_TEST_SUITE_P(workItemOrSubgroupTest, FFTTest,
                         ::testing::ConvertGenerator<param_tuple>(::testing::Combine(
                             ::testing::Values(1, 3, 555), ::testing::Values(16, 24, 27, 32, 48, 56))));
// sizes that use subgroup implementation
INSTANTIATE_TEST_SUITE_P(SubgroupTest, FFTTest,
                         ::testing::ConvertGenerator<param_tuple>(
                             ::testing::Combine(::testing::Values(1, 3, 555), ::testing::Values(64, 65, 84, 91, 104))));
// Backward FFT test suite
INSTANTIATE_TEST_SUITE_P(BackwardFFT, BwdTest,
                         ::testing::ConvertGenerator<param_tuple>(::testing::Combine(::testing::Values(1, 555),
                                                                                     ::testing::Values(14, 48, 104))));

#endif
