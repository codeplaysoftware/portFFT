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

// Mandatory parameters: placement, layout, direction, batch, lengths
// Optional parameters: [forward_scale, backward_scale]
class FFTTest : public ::testing::TestWithParam<test_params> {};
class InvalidFFTTest : public ::testing::TestWithParam<test_params> {};

using sizes_t = std::vector<std::size_t>;

constexpr test_placement_layouts_params valid_placement_layouts[] = {
    {placement::IN_PLACE, detail::layout::PACKED, detail::layout::PACKED},
    {placement::IN_PLACE, detail::layout::BATCH_INTERLEAVED, detail::layout::BATCH_INTERLEAVED},
#ifdef PORTFFT_ENABLE_OOP_BUILDS
    {placement::OUT_OF_PLACE, detail::layout::PACKED, detail::layout::PACKED},
    {placement::OUT_OF_PLACE, detail::layout::PACKED, detail::layout::BATCH_INTERLEAVED},
#endif
    {placement::OUT_OF_PLACE, detail::layout::BATCH_INTERLEAVED, detail::layout::BATCH_INTERLEAVED},
    {placement::OUT_OF_PLACE, detail::layout::BATCH_INTERLEAVED, detail::layout::PACKED}};
auto all_valid_placement_layouts = ::testing::ValuesIn(valid_placement_layouts);

constexpr test_placement_layouts_params valid_oop_placement_layouts[] = {
    {placement::OUT_OF_PLACE, detail::layout::PACKED, detail::layout::PACKED},
    {placement::OUT_OF_PLACE, detail::layout::PACKED, detail::layout::BATCH_INTERLEAVED},
    {placement::OUT_OF_PLACE, detail::layout::BATCH_INTERLEAVED, detail::layout::BATCH_INTERLEAVED},
    {placement::OUT_OF_PLACE, detail::layout::BATCH_INTERLEAVED, detail::layout::PACKED}};
auto all_valid_oop_placement_layouts = ::testing::ValuesIn(valid_oop_placement_layouts);

constexpr test_placement_layouts_params valid_multi_dim_placement_layouts[] = {
    {placement::IN_PLACE, detail::layout::PACKED, detail::layout::PACKED}
#ifdef PORTFFT_ENABLE_OOP_BUILDS
    ,
    {placement::OUT_OF_PLACE, detail::layout::PACKED, detail::layout::PACKED}
#endif
};
auto all_valid_multi_dim_placement_layouts = ::testing::ValuesIn(valid_multi_dim_placement_layouts);

auto ip_packed_layout = ::testing::Values(
    test_placement_layouts_params{placement::IN_PLACE, detail::layout::PACKED, detail::layout::PACKED});
auto ip_batch_interleaved_layout = ::testing::Values(test_placement_layouts_params{
    placement::IN_PLACE, detail::layout::BATCH_INTERLEAVED, detail::layout::BATCH_INTERLEAVED});
auto ip_unpacked_unpacked_layout = ::testing::Values(
    test_placement_layouts_params{placement::IN_PLACE, detail::layout::UNPACKED, detail::layout::UNPACKED});

auto oop_packed_packed_layout = ::testing::Values(
    test_placement_layouts_params{placement::OUT_OF_PLACE, detail::layout::PACKED, detail::layout::PACKED});
auto oop_unpacked_unpacked_layout = ::testing::Values(
    test_placement_layouts_params{placement::OUT_OF_PLACE, detail::layout::UNPACKED, detail::layout::UNPACKED});

auto all_unpacked_unpacked_layout = ::testing::Values(
    test_placement_layouts_params{placement::IN_PLACE, detail::layout::UNPACKED, detail::layout::UNPACKED},
    test_placement_layouts_params{placement::OUT_OF_PLACE, detail::layout::UNPACKED, detail::layout::UNPACKED});

constexpr test_placement_layouts_params valid_global_layouts[] = {
#ifdef PORTFFT_ENABLE_OOP_BUILDS
    {placement::OUT_OF_PLACE, detail::layout::PACKED, detail::layout::PACKED},
#endif
    {placement::IN_PLACE, detail::layout::PACKED, detail::layout::PACKED}};
auto all_valid_global_placement_layouts = ::testing::ValuesIn(valid_global_layouts);

auto fwd_only = ::testing::Values(direction::FORWARD);
auto bwd_only = ::testing::Values(direction::BACKWARD);
auto both_directions = ::testing::Values(direction::FORWARD, direction::BACKWARD);

auto complex_storages = ::testing::Values(complex_storage::INTERLEAVED_COMPLEX, complex_storage::SPLIT_COMPLEX);
auto interleaved_storage = ::testing::Values(complex_storage::INTERLEAVED_COMPLEX);

// sizes that use workitem implementation
INSTANTIATE_TEST_SUITE_P(workItemTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_placement_layouts, fwd_only, complex_storages, ::testing::Values(1, 3, 33000),
                             ::testing::Values(sizes_t{1}, sizes_t{2}, sizes_t{3}, sizes_t{4}, sizes_t{8}))),
                         test_params_print());
// sizes that might use workitem or subgroup implementation depending on device
// and configurations
INSTANTIATE_TEST_SUITE_P(workItemOrSubgroupTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_placement_layouts, fwd_only, complex_storages, ::testing::Values(1, 3, 555),
                             ::testing::Values(sizes_t{16}, sizes_t{32}))),
                         test_params_print());
// sizes that use subgroup implementation
INSTANTIATE_TEST_SUITE_P(SubgroupTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_placement_layouts, fwd_only, complex_storages, ::testing::Values(1, 3, 555),
                             ::testing::Values(sizes_t{64}, sizes_t{96}, sizes_t{128}))),
                         test_params_print());
// sizes that use subgroup implementation
INSTANTIATE_TEST_SUITE_P(SubgroupRegressionTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             ip_batch_interleaved_layout, fwd_only, interleaved_storage, ::testing::Values(44, 100),
                             ::testing::Values(sizes_t{80}, sizes_t{100}))),
                         test_params_print());
// sizes that might use subgroup or workgroup implementation depending on device
// and configurations
INSTANTIATE_TEST_SUITE_P(SubgroupOrWorkgroupTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_placement_layouts, fwd_only, complex_storages, ::testing::Values(1, 131),
                             ::testing::Values(sizes_t{256}, sizes_t{512}, sizes_t{1024}))),
                         test_params_print());
// Regression test where subgroup or workgroup implemention depended on correct local memory requirement calcs.
INSTANTIATE_TEST_SUITE_P(SubgroupOrWorkgroupRegressionTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(
                             ::testing::Combine(ip_packed_layout, fwd_only, interleaved_storage,
                                                ::testing::Values(1, 131), ::testing::Values(sizes_t{1536}))),
                         test_params_print());
// sizes that use workgroup implementation
INSTANTIATE_TEST_SUITE_P(WorkgroupTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_placement_layouts, fwd_only, complex_storages, ::testing::Values(1, 3),
                             ::testing::Values(sizes_t{2048}, sizes_t{3072}, sizes_t{4096}))),
                         test_params_print());

// Sizes that can use either workgroup or Global implementation
INSTANTIATE_TEST_SUITE_P(WorkgroupOrGlobal, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_global_placement_layouts, fwd_only, complex_storages, ::testing::Values(1, 128),
                             ::testing::Values(sizes_t{8192}, sizes_t{16384}))),
                         test_params_print());

// Sizes that use the global implementations
INSTANTIATE_TEST_SUITE_P(GlobalTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_global_placement_layouts, fwd_only, complex_storages, ::testing::Values(1, 3),
                             ::testing::Values(sizes_t{32768}, sizes_t{65536}, sizes_t{131072}))),
                         test_params_print());

INSTANTIATE_TEST_SUITE_P(WorkgroupOrGlobalRegressionTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(
                             ::testing::Combine(ip_packed_layout, fwd_only, interleaved_storage, ::testing::Values(3),
                                                ::testing::Values(sizes_t{9800}, sizes_t{15360}, sizes_t{68640}))),
                         test_params_print());

// Test suite contains both Prime sized values(211, 523, 65537) as well as the sizes which have prime factors that we
// cannot handle (33012, 45232)
INSTANTIATE_TEST_SUITE_P(PrimeSizedTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_global_placement_layouts, fwd_only, interleaved_storage, ::testing::Values(1, 3),
                             ::testing::Values(sizes_t{211}, sizes_t{523}, sizes_t{65537}, sizes_t{33012},
                                               sizes_t{45232}))),
                         test_params_print());

// Backward FFT test suite
INSTANTIATE_TEST_SUITE_P(BackwardTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_placement_layouts, bwd_only, complex_storages, ::testing::Values(1, 3),
                             ::testing::Values(sizes_t{8}, sizes_t{9}, sizes_t{16}, sizes_t{32}, sizes_t{64},
                                               sizes_t{4096}))),
                         test_params_print());

// Backward FFT test suite
// TODO: move these into the BackwardTest once the global impl supports strided layout
INSTANTIATE_TEST_SUITE_P(BackwardGlobalTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_global_placement_layouts, bwd_only, complex_storages, ::testing::Values(1, 3),
                             ::testing::Values(sizes_t{32768}, sizes_t{65536}))),
                         test_params_print());

// Multidimensional FFT test suite
INSTANTIATE_TEST_SUITE_P(MultidimensionalTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_multi_dim_placement_layouts, both_directions, complex_storages,
                             ::testing::Values(1, 3),
                             ::testing::Values(sizes_t{2, 4}, sizes_t{4, 2}, sizes_t{16, 512}, sizes_t{64, 2048},
                                               sizes_t{2, 3, 6}, sizes_t{2, 3, 2, 3}))),
                         test_params_print());

// Offset data test suite

// Pairs of offsets: {forward_offset, backward_offset}
constexpr std::pair<std::size_t, std::size_t> matched_offset_values[] = {{8, 8}, {67, 67}};
auto matched_offsets = ::testing::ValuesIn(matched_offset_values);
constexpr std::pair<std::size_t, std::size_t> mismatched_offset_values[] = {{0, 2049}, {2049, 0}, {2047, 2049}};
auto mismatched_offsets = ::testing::ValuesIn(mismatched_offset_values);

INSTANTIATE_TEST_SUITE_P(OffsetsMatchedTest, FFTTest,
                         ::testing::ConvertGenerator<offsets_param_tuple>(::testing::Combine(
                             all_valid_placement_layouts, fwd_only, interleaved_storage, ::testing::Values(33),
                             ::testing::Values(sizes_t{2048}), matched_offsets)),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(OffsetsMultiDimensionalTest, FFTTest,
                         ::testing::ConvertGenerator<offsets_param_tuple>(::testing::Combine(
                             all_valid_multi_dim_placement_layouts, fwd_only, interleaved_storage,
                             ::testing::Values(33), ::testing::Values(sizes_t{16, 512}), matched_offsets)),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(OffsetsMismatchedTest, FFTTest,
                         ::testing::ConvertGenerator<offsets_param_tuple>(::testing::Combine(
                             all_valid_oop_placement_layouts, both_directions, interleaved_storage,
                             ::testing::Values(33), ::testing::Values(sizes_t{2048}), mismatched_offsets)),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(OffsetsWIErrorRegressionTest, FFTTest,
                         ::testing::ConvertGenerator<offsets_param_tuple>(::testing::Combine(
                             all_valid_oop_placement_layouts, both_directions, interleaved_storage,
                             ::testing::Values(33000), ::testing::Values(sizes_t{8}), mismatched_offsets)),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(OffsetsMDErrorRegressionTest, FFTTest,
                         ::testing::ConvertGenerator<offsets_param_tuple>(::testing::Combine(
                             ::testing::Values(test_placement_layouts_params{
                                 placement::OUT_OF_PLACE, detail::layout::PACKED, detail::layout::PACKED}),
                             fwd_only, interleaved_storage, ::testing::Values(2), ::testing::Values(sizes_t{4, 4}),
                             ::testing::Values(std::pair<std::size_t, std::size_t>({2, 0})))),
                         test_params_print());

// Scaled FFTs test suite
auto scales = ::testing::Values(-1.0, 2.0);
INSTANTIATE_TEST_SUITE_P(FwdScaledFFTTest, FFTTest,
                         ::testing::ConvertGenerator<scales_param_tuple>(::testing::Combine(
                             oop_packed_packed_layout, fwd_only, interleaved_storage, ::testing::Values(3),
                             ::testing::Values(sizes_t{9}, sizes_t{16}, sizes_t{64}, sizes_t{512}, sizes_t{4096},
                                               sizes_t{16, 512}),
                             scales, ::testing::Values(1.0))),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(BwdScaledFFTTest, FFTTest,
                         ::testing::ConvertGenerator<scales_param_tuple>(::testing::Combine(
                             oop_packed_packed_layout, bwd_only, interleaved_storage, ::testing::Values(3),
                             ::testing::Values(sizes_t{9}, sizes_t{16}, sizes_t{64}, sizes_t{512}, sizes_t{4096},
                                               sizes_t{16, 512}),
                             ::testing::Values(1.0), scales)),
                         test_params_print());

INSTANTIATE_TEST_SUITE_P(workItemStridedOOPInOrder, FFTTest,
                         ::testing::ConvertGenerator<layout_param_tuple>(::testing::Combine(
                             oop_unpacked_unpacked_layout, both_directions, complex_storages,
                             ::testing::Values(1, 3, 33000ul),
                             ::testing::Values(layout_params{{3}, {4}, {7}}, layout_params{{8}, {11}, {2}},
                                               layout_params{{9}, {3}, {4}, 30, 40}))),
                         test_params_print());
// The LikeBatchInterleaved tests must have stride >= number of transforms
INSTANTIATE_TEST_SUITE_P(
    workItemStridedOOPLikeBatchInterleaved, FFTTest,
    ::testing::ConvertGenerator<layout_param_tuple>(::testing::Combine(
        oop_unpacked_unpacked_layout, both_directions, complex_storages, ::testing::Values(1, 10, 33),
        ::testing::Values(layout_params{{8}, {33}, {99}, 1, 3}, layout_params{{8}, {33}, {2}, 1, 16},
                          layout_params{{8}, {2}, {66}, 16, 2}))),
    test_params_print());
INSTANTIATE_TEST_SUITE_P(workItemStridedIP, FFTTest,
                         ::testing::ConvertGenerator<layout_param_tuple>(::testing::Combine(
                             ip_unpacked_unpacked_layout, both_directions, complex_storages,
                             ::testing::Values(1, 3, 33000ul),
                             ::testing::Values(layout_params{{3}, {4}, {4}}, layout_params{{9}, {3}, {3}, 25, 25}))),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(
    workItemStridedLikeBatchInterleaved, FFTTest,
    ::testing::ConvertGenerator<layout_param_tuple>(::testing::Combine(
        ip_unpacked_unpacked_layout, both_directions, complex_storages, ::testing::Values(1, 3, 33),
        ::testing::Values(layout_params{{3}, {66}, {66}, 2, 2}, layout_params{{6}, {40}, {40}, 1, 1}))),
    test_params_print());
// these layouts are only valid because there is only a single batch
INSTANTIATE_TEST_SUITE_P(WorkItemStridedStrideEqualsDistance, FFTTest,
                         ::testing::ConvertGenerator<layout_param_tuple>(::testing::Combine(
                             all_unpacked_unpacked_layout, both_directions, complex_storages, ::testing::Values(1),
                             ::testing::Values(layout_params{{8}, {2}, {2}, 2, 2},
                                               layout_params{{8}, {1}, {1}, 1, 1}))),
                         test_params_print());

// clang-format off
// Arbitrary interleaved FFT test suites
// The strides and distances are set so that no elements overlap but there are no single continuous dimension in memory either.
// This configuration is impractical but technically valid. For instance for n_batches=4, fft_size=4, stride=4, distance=3:
// Index in memory:     0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   20   21
// Batch and FFT index: b0i0           b1i0 b0i1      b2i0 b1i1 b0i2 b3i0 b2i1 b1i2 b0i3 b3i1 b2i2 b1i3      b3i2 b2i3           b3i3
// clang-format on
INSTANTIATE_TEST_SUITE_P(workItemStridedArbitraryInterleaved, FFTTest,
                         ::testing::ConvertGenerator<layout_param_tuple>(::testing::Combine(
                             all_unpacked_unpacked_layout, both_directions, complex_storages, ::testing::Values(4),
                             ::testing::Values(layout_params{{4}, {4}, {4}, 3, 3}))),
                         test_params_print());

// Invalid configurations test suite
INSTANTIATE_TEST_SUITE_P(InvalidLength, InvalidFFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(
                             ::testing::Combine(all_valid_placement_layouts, both_directions, complex_storages,
                                                ::testing::Values(1), ::testing::Values(0))),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(InvalidBatch, InvalidFFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(
                             ::testing::Combine(all_valid_placement_layouts, both_directions, complex_storages,
                                                ::testing::Values(0), ::testing::Values(1))),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(InvalidDistance, InvalidFFTTest,
                         ::testing::ConvertGenerator<layout_param_tuple>(::testing::Combine(
                             oop_unpacked_unpacked_layout, both_directions, complex_storages, ::testing::Values(2),
                             ::testing::Values(layout_params{{5}, {5}, {1}, 0, 5},
                                               layout_params{{5}, {1}, {5}, 5, 0}))),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(InvalidNonPositiveStrides, InvalidFFTTest,
                         ::testing::ConvertGenerator<layout_param_tuple>(::testing::Combine(
                             oop_unpacked_unpacked_layout, both_directions, complex_storages, ::testing::Values(1),
                             ::testing::Values(layout_params{{5}, {0}, {1}}, layout_params{{5}, {1}, {0}},
                                               layout_params{{5, 12}, {12, 1}, {12, 0}}))),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(InvalidShortDistance, InvalidFFTTest,
                         ::testing::ConvertGenerator<layout_param_tuple>(::testing::Combine(
                             oop_unpacked_unpacked_layout, both_directions, complex_storages, ::testing::Values(2),
                             ::testing::Values(layout_params{{8}, {1}, {1}, 7, 8},
                                               layout_params{{8, 4}, {8, 2}, {4, 1}, 24, 24}))),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(InvalidIPNotMatchingStridesDistance, InvalidFFTTest,
                         ::testing::ConvertGenerator<layout_param_tuple>(::testing::Combine(
                             ip_unpacked_unpacked_layout, both_directions, complex_storages, ::testing::Values(2),
                             ::testing::Values(layout_params{{8}, {2}, {1}, 16, 8},
                                               layout_params{{8, 4}, {8, 2}, {8, 2}, 48, 50}))),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(InvalidOverlap, InvalidFFTTest,
                         ::testing::ConvertGenerator<layout_param_tuple>(::testing::Combine(
                             oop_unpacked_unpacked_layout, both_directions, complex_storages, ::testing::Values(3),
                             ::testing::Values(layout_params{{4}, {1}, {1}, 1, 4},
                                               layout_params{{4}, {1}, {2}, 4, 3}))),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(InvalidOverlapLarge, InvalidFFTTest,
                         ::testing::ConvertGenerator<layout_param_tuple>(
                             ::testing::Combine(oop_unpacked_unpacked_layout, both_directions, complex_storages,
                                                ::testing::Values(3333334),
                                                ::testing::Values(layout_params{{8}, {3333333}, {3333333}, 1, 1}))),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(InvalidStrideEqualsDistance, InvalidFFTTest,
                         ::testing::ConvertGenerator<layout_param_tuple>(::testing::Combine(
                             oop_unpacked_unpacked_layout, both_directions, complex_storages, ::testing::Values(2),
                             ::testing::Values(layout_params{{8}, {2}, {2}, 2, 2},
                                               layout_params{{8}, {1}, {1}, 1, 1}))),
                         test_params_print());

#define INSTANTIATE_TESTS_FULL(TYPE, MEMORY)                                                                        \
  TEST_P(FFTTest, TYPE##_##MEMORY##_C2C) {                                                                          \
    auto params = GetParam();                                                                                       \
    if (params.dir == portfft::direction::FORWARD) {                                                                \
      if (params.storage == portfft::complex_storage::INTERLEAVED_COMPLEX) {                                        \
        run_test<test_memory::MEMORY, TYPE, portfft::direction::FORWARD,                                            \
                 portfft::complex_storage::INTERLEAVED_COMPLEX>(params);                                            \
      } else {                                                                                                      \
        run_test<test_memory::MEMORY, TYPE, portfft::direction::FORWARD, portfft::complex_storage::SPLIT_COMPLEX>(  \
            params);                                                                                                \
      }                                                                                                             \
    } else {                                                                                                        \
      if (params.storage == portfft::complex_storage::INTERLEAVED_COMPLEX) {                                        \
        run_test<test_memory::MEMORY, TYPE, portfft::direction::BACKWARD,                                           \
                 portfft::complex_storage::INTERLEAVED_COMPLEX>(params);                                            \
      } else {                                                                                                      \
        run_test<test_memory::MEMORY, TYPE, portfft::direction::BACKWARD, portfft::complex_storage::SPLIT_COMPLEX>( \
            params);                                                                                                \
      }                                                                                                             \
    }                                                                                                               \
  }

#ifdef PORTFFT_ENABLE_BUFFER_BUILDS
#define INSTANTIATE_TESTS(TYPE)     \
  INSTANTIATE_TESTS_FULL(TYPE, usm) \
  INSTANTIATE_TESTS_FULL(TYPE, buffer)
#else
#define INSTANTIATE_TESTS(TYPE) INSTANTIATE_TESTS_FULL(TYPE, usm)
#endif

// The result of this test should not be dependent on scalar type or memory type
TEST_P(InvalidFFTTest, Test) {
  auto params = GetParam();
  sycl::queue queue;
  auto desc = get_descriptor<float>(params);
  EXPECT_THROW(desc.commit(queue), portfft::invalid_configuration);
}

#endif
