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

#include <type_traits>

#include <gtest/gtest.h>

#include "fft_test_utils.hpp"

// Parameters in [] are optional:
// layout, batch, length, [fwd_strides, bwd_strides], [fwd_distance, bwd_distance]
class FFTTest : public ::testing::TestWithParam<test_params> {};
class InvalidFFTTest : public ::testing::TestWithParam<test_params> {};

// Test suites must match with the configurations in scripts/test_reference_specification.py
// to generate the expected results.
// The global variables have a prefix based on the instantiation name to avoid conflicting names (i.e.
// DistanceWorkItemTest -> dwi_).

auto ip_and_oop = ::testing::Values(placement::IN_PLACE, placement::OUT_OF_PLACE);
auto ip = ::testing::Values(placement::IN_PLACE);
auto oop = ::testing::Values(placement::OUT_OF_PLACE);
auto packed_and_transposed = ::testing::Values(detail::layout::PACKED, detail::layout::BATCH_INTERLEAVED);
auto fwd_and_bwd = ::testing::Values(direction::FORWARD, direction::BACKWARD);
auto fwd_only = ::testing::Values(direction::FORWARD);
auto bwd_only = ::testing::Values(direction::BACKWARD);

// sizes that use workitem implementation
INSTANTIATE_TEST_SUITE_P(workItemTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(
                             ::testing::Combine(ip_and_oop, packed_and_transposed, fwd_and_bwd,
                                                ::testing::Values(1, 3, 33000), ::testing::Values(1, 2, 3, 4, 8))),
                         test_params_print());
// sizes that might use workitem or subgroup implementation depending on device
// and configuration
INSTANTIATE_TEST_SUITE_P(workItemOrSubgroupTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(
                             ::testing::Combine(ip_and_oop, packed_and_transposed, fwd_and_bwd,
                                                ::testing::Values(1, 3, 555), ::testing::Values(16, 32))),
                         test_params_print());
// sizes that use subgroup implementation
INSTANTIATE_TEST_SUITE_P(SubgroupTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(
                             ::testing::Combine(ip_and_oop, packed_and_transposed, fwd_and_bwd,
                                                ::testing::Values(1, 3, 555), ::testing::Values(64, 96, 128))),
                         test_params_print());

INSTANTIATE_TEST_SUITE_P(SubgroupOrWorkgroupTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(
                             ::testing::Combine(ip_and_oop, packed_and_transposed, fwd_and_bwd, ::testing::Values(1, 3),
                                                ::testing::Values(256, 512, 1024))),
                         test_params_print());

INSTANTIATE_TEST_SUITE_P(WorkgroupTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(
                             ::testing::Combine(ip_and_oop, packed_and_transposed, fwd_and_bwd, ::testing::Values(1, 3),
                                                ::testing::Values(2048, 3072, 4096))),
                         test_params_print());

// Backward FFT test suite
INSTANTIATE_TEST_SUITE_P(BackwardTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(
                             ::testing::Combine(ip_and_oop, packed_and_transposed, fwd_and_bwd, ::testing::Values(1),
                                                ::testing::Values(16, 64, 2048))),
                         test_params_print());

// Strided FFT test suite
// stride_param_tuple sets a default distance so that each FFTs are contiguous
using strides_t = std::vector<std::size_t>;
auto s_strides = ::testing::Values(strides_t{0, 2}, strides_t{0, 64}, strides_t{5, 1}, strides_t{3, 5});
INSTANTIATE_TEST_SUITE_P(StridedTest, FFTTest,
                         ::testing::ConvertGenerator<stride_param_tuple>(
                             ::testing::Combine(oop, fwd_and_bwd, ::testing::Values(1, 3),
                                                ::testing::Values(4, 128, 4096), s_strides, s_strides)),
                         test_params_print());

// Distance FFT test suites
auto d_batch = ::testing::Values(2, 50);
auto dwi_distance = ::testing::Values(5, 9);
INSTANTIATE_TEST_SUITE_P(DistanceWorkItemTest, FFTTest,
                         ::testing::ConvertGenerator<distance_param_tuple>(::testing::Combine(
                             oop, fwd_and_bwd, d_batch, ::testing::Values(4), dwi_distance, dwi_distance)),
                         test_params_print());
auto dsg_distance = ::testing::Values(129, 200);
INSTANTIATE_TEST_SUITE_P(DistanceSubgroupTest, FFTTest,
                         ::testing::ConvertGenerator<distance_param_tuple>(::testing::Combine(
                             oop, fwd_and_bwd, d_batch, ::testing::Values(128), dsg_distance, dsg_distance)),
                         test_params_print());
auto dwg_distance = ::testing::Values(4097, 4100);
INSTANTIATE_TEST_SUITE_P(DistanceWorkgroupTest, FFTTest,
                         ::testing::ConvertGenerator<distance_param_tuple>(::testing::Combine(
                             oop, fwd_and_bwd, d_batch, ::testing::Values(4096), dwg_distance, dwg_distance)),
                         test_params_print());

// Strided and distance FFT test suites
// Test each implementation level, and with a large stride and distance for both direction (`BothDir`) or for only one
// of the direction (`Fwd` or `Bwd`)
auto sd_default_strides = ::testing::Values(strides_t{0, 1});
auto sdwi_batch = ::testing::Values(5);
auto sdwi_length = ::testing::Values(16);
auto sdwi_strides = ::testing::Values(strides_t{0, 5});
auto sdwi_distance = ::testing::Values(5 * 16 + 3);
auto sdwi_default_distance = sdwi_length;
INSTANTIATE_TEST_SUITE_P(StridedDistanceWorkItemBothDirTest, FFTTest,
                         ::testing::ConvertGenerator<stride_distance_param_tuple>(
                             ::testing::Combine(ip_and_oop, fwd_and_bwd, sdwi_batch, sdwi_length, sdwi_strides,
                                                sdwi_strides, sdwi_distance, sdwi_distance)),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(StridedDistanceWorkItemFwdTest, FFTTest,
                         ::testing::ConvertGenerator<stride_distance_param_tuple>(
                             ::testing::Combine(oop, fwd_only, sdwi_batch, sdwi_length, sdwi_strides,
                                                sd_default_strides, sdwi_distance, sdwi_default_distance)),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(StridedDistanceWorkItemBwdTest, FFTTest,
                         ::testing::ConvertGenerator<stride_distance_param_tuple>(
                             ::testing::Combine(oop, bwd_only, sdwi_batch, sdwi_length, sd_default_strides,
                                                sdwi_strides, sdwi_default_distance, sdwi_distance)),
                         test_params_print());
auto sdsg_batch = ::testing::Values(3);
auto sdsg_length = ::testing::Values(128);
auto sdsg_strides = ::testing::Values(strides_t{0, 3});
auto sdsg_distance = ::testing::Values(3 * 128 + 3);
auto sdsg_default_distance = sdsg_length;
INSTANTIATE_TEST_SUITE_P(StridedDistanceSubgroupBothDirTest, FFTTest,
                         ::testing::ConvertGenerator<stride_distance_param_tuple>(
                             ::testing::Combine(ip_and_oop, fwd_and_bwd, sdsg_batch, sdsg_length, sdsg_strides,
                                                sdsg_strides, sdsg_distance, sdsg_distance)),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(StridedDistanceSubgroupFwdTest, FFTTest,
                         ::testing::ConvertGenerator<stride_distance_param_tuple>(
                             ::testing::Combine(oop, fwd_only, sdsg_batch, sdsg_length, sdsg_strides,
                                                sd_default_strides, sdsg_distance, sdsg_default_distance)),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(StridedDistanceSubgroupBwdTest, FFTTest,
                         ::testing::ConvertGenerator<stride_distance_param_tuple>(
                             ::testing::Combine(oop, bwd_only, sdsg_batch, sdsg_length, sd_default_strides,
                                                sdsg_strides, sdsg_default_distance, sdsg_distance)),
                         test_params_print());
auto sdwg_batch = ::testing::Values(2);
auto sdwg_length = ::testing::Values(4096);
auto sdwg_strides = ::testing::Values(strides_t{0, 2});
auto sdwg_distance = ::testing::Values(2 * 4096 + 1);
auto sdwg_default_distance = sdwg_length;
INSTANTIATE_TEST_SUITE_P(StridedDistanceWorkgroupBothDirTest, FFTTest,
                         ::testing::ConvertGenerator<stride_distance_param_tuple>(
                             ::testing::Combine(ip_and_oop, fwd_and_bwd, sdwg_batch, sdwg_length, sdwg_strides,
                                                sdwg_strides, sdwg_distance, sdwg_distance)),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(StridedDistanceWorkgroupFwdTest, FFTTest,
                         ::testing::ConvertGenerator<stride_distance_param_tuple>(
                             ::testing::Combine(oop, fwd_only, sdwg_batch, sdwg_length, sdwg_strides,
                                                sd_default_strides, sdwg_distance, sdwg_default_distance)),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(StridedDistanceWorkgroupBwdTest, FFTTest,
                         ::testing::ConvertGenerator<stride_distance_param_tuple>(
                             ::testing::Combine(oop, bwd_only, sdwg_batch, sdwg_length, sd_default_strides,
                                                sdwg_strides, sdwg_default_distance, sdwg_distance)),
                         test_params_print());

// Batch interleaved FFT test suites
// The distance is smaller than the size of an FFT and the stride is larger or equal to the batch size.
// In this layout the batch dimension is the inner-most dimension.
auto bi_batch = ::testing::Values(3);
auto bi_strides = ::testing::Values(strides_t{0, 3});
auto bi_distance = ::testing::Values(1);
INSTANTIATE_TEST_SUITE_P(BatchInterleavedWorkItemTest, FFTTest,
                         ::testing::ConvertGenerator<stride_distance_param_tuple>(
                             ::testing::Combine(ip_and_oop, fwd_and_bwd, bi_batch, ::testing::Values(16), bi_strides,
                                                bi_strides, bi_distance, bi_distance)),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(BatchInterleavedSubgroupTest, FFTTest,
                         ::testing::ConvertGenerator<stride_distance_param_tuple>(
                             ::testing::Combine(ip_and_oop, fwd_and_bwd, bi_batch, ::testing::Values(128), bi_strides,
                                                bi_strides, bi_distance, bi_distance)),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(BatchInterleavedWorkgroupTest, FFTTest,
                         ::testing::ConvertGenerator<stride_distance_param_tuple>(
                             ::testing::Combine(ip_and_oop, fwd_and_bwd, bi_batch, ::testing::Values(4096), bi_strides,
                                                bi_strides, bi_distance, bi_distance)),
                         test_params_print());
auto bils_strides = ::testing::Values(strides_t{0, 60});
auto bils_distance = ::testing::Values(3);
INSTANTIATE_TEST_SUITE_P(BatchInterleavedLargerStrideDistanceTest, FFTTest,
                         ::testing::ConvertGenerator<stride_distance_param_tuple>(
                             ::testing::Combine(ip_and_oop, fwd_and_bwd, ::testing::Values(20), ::testing::Values(16),
                                                bils_strides, bils_strides, bils_distance, bils_distance)),
                         test_params_print());

// clang-format off
// Arbitrary interleaved FFT test suites
// The strides and distances are set so that no elements overlap but there are no single continuous dimension in memory either.
// This configuration is impractical but technically valid. For instance for n_batches=4, fft_size=4, stride=4, distance=3:
// Index in memory:     0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   20   21
// Batch and FFT index: b0i0           b1i0 b0i1      b2i0 b1i1 b0i2 b3i0 b2i1 b1i2 b0i3 b3i1 b2i2 b1i3      b3i2 b2i3           b3i3
// clang-format on
auto ai_strides = ::testing::Values(strides_t{0, 4});
auto ai_distance = ::testing::Values(3);
INSTANTIATE_TEST_SUITE_P(ArbitraryInterleavedTest, FFTTest,
                         ::testing::ConvertGenerator<stride_distance_param_tuple>(
                             ::testing::Combine(ip_and_oop, fwd_and_bwd, ::testing::Values(4), ::testing::Values(4),
                                                ai_strides, ai_strides, ai_distance, ai_distance)),
                         test_params_print());

// Read overlap test suites
// Test that it is allowed to read the same elements multiple times.
INSTANTIATE_TEST_SUITE_P(OverlapReadFwdTest, FFTTest,
                         ::testing::ConvertGenerator<distance_param_tuple>(
                             ::testing::Combine(oop, fwd_only, ::testing::Values(3), ::testing::Values(4),
                                                ::testing::Values(1), ::testing::Values(4))),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(OverlapReadBwdTest, FFTTest,
                         ::testing::ConvertGenerator<distance_param_tuple>(
                             ::testing::Combine(oop, bwd_only, ::testing::Values(3), ::testing::Values(4),
                                                ::testing::Values(4), ::testing::Values(1))),
                         test_params_print());

// Invalid configuration test suites
auto unpacked = ::testing::Values(detail::layout::UNPACKED);
auto zero = ::testing::Values(0);
auto one = ::testing::Values(1);
INSTANTIATE_TEST_SUITE_P(InvalidLengthTest, InvalidFFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(ip_and_oop, unpacked,
                                                                                           fwd_and_bwd, one, zero)),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(InvalidBatchTest, InvalidFFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(ip_and_oop, unpacked,
                                                                                           fwd_and_bwd, zero, one)),
                         test_params_print());
auto invalid_strides = ::testing::Values(strides_t{0}, strides_t{0, 0}, strides_t{0, 1, 1});
auto valid_strides = ::testing::Values(strides_t{0, 1});
INSTANTIATE_TEST_SUITE_P(InvalidFwdStridesTest, InvalidFFTTest,
                         ::testing::ConvertGenerator<stride_param_tuple>(
                             ::testing::Combine(oop, fwd_and_bwd, one, one, invalid_strides, valid_strides)),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(InvalidBwdStridesTest, InvalidFFTTest,
                         ::testing::ConvertGenerator<stride_param_tuple>(
                             ::testing::Combine(oop, fwd_and_bwd, one, one, valid_strides, invalid_strides)),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(InvalidMismatchStridesInplaceTest, InvalidFFTTest,
                         ::testing::ConvertGenerator<stride_param_tuple>(::testing::Combine(
                             ip, fwd_and_bwd, one, one, valid_strides, ::testing::Values(strides_t{0, 2}))),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(InvalidFwdDistanceTest, InvalidFFTTest,
                         ::testing::ConvertGenerator<distance_param_tuple>(
                             ::testing::Combine(oop, fwd_and_bwd, ::testing::Values(2), one, zero, one)),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(InvalidBwdDistanceTest, InvalidFFTTest,
                         ::testing::ConvertGenerator<distance_param_tuple>(
                             ::testing::Combine(oop, fwd_and_bwd, ::testing::Values(2), one, one, zero)),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(InvalidMismatchDistanceInplaceTest, InvalidFFTTest,
                         ::testing::ConvertGenerator<distance_param_tuple>(
                             ::testing::Combine(ip, fwd_and_bwd, ::testing::Values(2), one, one, ::testing::Values(2))),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(InvalidSmallDistanceTest, InvalidFFTTest,
                         ::testing::ConvertGenerator<distance_param_tuple>(
                             ::testing::Combine(oop, fwd_and_bwd, ::testing::Values(2), ::testing::Values(4),
                                                ::testing::Values(3), ::testing::Values(3))),
                         test_params_print());
auto iss_strides = ::testing::Values(strides_t{0, 5});
auto iss_distance = ::testing::Values(1);
INSTANTIATE_TEST_SUITE_P(InvalidSmallStrideTest, InvalidFFTTest,
                         ::testing::ConvertGenerator<stride_distance_param_tuple>(
                             ::testing::Combine(ip_and_oop, fwd_and_bwd, ::testing::Values(6), ::testing::Values(3),
                                                iss_strides, iss_strides, iss_distance, iss_distance)),
                         test_params_print());

template <typename DeviceSelector>
std::pair<std::optional<sycl::queue>, std::string> get_queue(DeviceSelector selector) {
  try {
    sycl::queue queue(selector);
    return std::make_pair(queue, "");
  } catch (sycl::exception& e) {
    return std::make_pair(std::nullopt, e.what());
  }
}

int fp64_selector(sycl::device dev) {
  if (dev.has(sycl::aspect::fp64))
    return 1;
  else
    return -1;
}

#define CHECK_QUEUE(queue) \
  if (!queue.first) GTEST_SKIP() << queue.second;

#define INSTANTIATE_TESTS_FULL(TYPE, TYPE_NAME, MEM, MEM_NAME) \
  TEST_P(FFTTest, TYPE_NAME##_##MEM_NAME##_C2C) {              \
    auto param = GetParam();                                   \
    sycl::queue queue;                                         \
    if constexpr (std::is_same<TYPE, double>::value) {         \
      auto queue_pair = get_queue(fp64_selector);              \
      CHECK_QUEUE(queue_pair);                                 \
      queue = queue_pair.first.value();                        \
    }                                                          \
    check_fft_##MEM<TYPE>(param, queue);                       \
  }

#define INSTANTIATE_TESTS(TYPE, TYPE_NAME)          \
  INSTANTIATE_TESTS_FULL(TYPE, TYPE_NAME, usm, USM) \
  INSTANTIATE_TESTS_FULL(TYPE, TYPE_NAME, buffer, BUFFER)

// Test suites for expected invalid configurations.
// The compute functions are not called so the tests are not repeated for USM and buffer nor for double precision.
// The queue is still needed to commit.
#define INSTANTIATE_INVALID_TESTS()  \
  TEST_P(InvalidFFTTest, Test) {     \
    auto param = GetParam();         \
    sycl::queue queue;               \
    check_invalid_fft(param, queue); \
  }

#endif
