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

using sizes_t = std::vector<std::size_t>;

constexpr test_placement_layouts_params valid_placement_layouts[] = {
    {placement::IN_PLACE, detail::layout::PACKED, detail::layout::PACKED},
    {placement::IN_PLACE, detail::layout::BATCH_INTERLEAVED, detail::layout::BATCH_INTERLEAVED},
    {placement::OUT_OF_PLACE, detail::layout::PACKED, detail::layout::PACKED},
    {placement::OUT_OF_PLACE, detail::layout::PACKED, detail::layout::BATCH_INTERLEAVED},
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
    {placement::IN_PLACE, detail::layout::PACKED, detail::layout::PACKED},
    {placement::OUT_OF_PLACE, detail::layout::PACKED, detail::layout::PACKED}};
auto all_valid_multi_dim_placement_layouts = ::testing::ValuesIn(valid_multi_dim_placement_layouts);

auto oop_packed_layout = ::testing::Values(
    test_placement_layouts_params{placement::OUT_OF_PLACE, detail::layout::PACKED, detail::layout::PACKED});

constexpr test_placement_layouts_params valid_global_layouts[] = {
    {placement::OUT_OF_PLACE, detail::layout::PACKED, detail::layout::PACKED},
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
// sizes that might use subgroup or workgroup implementation depending on device
// and configurations
INSTANTIATE_TEST_SUITE_P(SubgroupOrWorkgroupTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_placement_layouts, fwd_only, interleaved_storage, ::testing::Values(1, 131),
                             ::testing::Values(sizes_t{256}, sizes_t{512}, sizes_t{1024}))),
                         test_params_print());
// sizes that use workgroup implementation
INSTANTIATE_TEST_SUITE_P(WorkgroupTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_placement_layouts, fwd_only, interleaved_storage, ::testing::Values(1, 3),
                             ::testing::Values(sizes_t{2048}, sizes_t{3072}, sizes_t{4096}))),
                         test_params_print());

// Sizes that can use either workgroup or Global implementation
INSTANTIATE_TEST_SUITE_P(WorkgroupOrGlobal, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_global_placement_layouts, fwd_only, interleaved_storage,
                             ::testing::Values(1, 128), ::testing::Values(sizes_t{8192}, sizes_t{16384}))),
                         test_params_print());

// Sizes that use the global implementations
INSTANTIATE_TEST_SUITE_P(GlobalTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_global_placement_layouts, fwd_only, interleaved_storage, ::testing::Values(1, 3),
                             ::testing::Values(sizes_t{32768}, sizes_t{65536}, sizes_t{131072}))),
                         test_params_print());

// Backward FFT test suite
INSTANTIATE_TEST_SUITE_P(BackwardTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_placement_layouts, bwd_only, interleaved_storage, ::testing::Values(1, 3),
                             ::testing::Values(sizes_t{8}, sizes_t{9}, sizes_t{16}, sizes_t{32}, sizes_t{64},
                                               sizes_t{4096}))),
                         test_params_print());

// Backward FFT test suite
// TODO: move these into the BackwardTest once the global impl supports strided layout
INSTANTIATE_TEST_SUITE_P(BackwardGlobalTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_global_placement_layouts, bwd_only, interleaved_storage, ::testing::Values(1, 3),
                             ::testing::Values(sizes_t{32768}, sizes_t{65536}))),
                         test_params_print());

// Multidimensional FFT test suite
INSTANTIATE_TEST_SUITE_P(MultidimensionalTest, FFTTest,
                         ::testing::ConvertGenerator<basic_param_tuple>(::testing::Combine(
                             all_valid_multi_dim_placement_layouts, both_directions, interleaved_storage,
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
                         ::testing::ConvertGenerator<scales_param_tuple>(
                             ::testing::Combine(oop_packed_layout, fwd_only, interleaved_storage, ::testing::Values(3),
                                                ::testing::Values(sizes_t{9}, sizes_t{16}, sizes_t{64}, sizes_t{512},
                                                                  sizes_t{4096}, sizes_t{16, 512}),
                                                scales, ::testing::Values(1.0))),
                         test_params_print());
INSTANTIATE_TEST_SUITE_P(BwdScaledFFTTest, FFTTest,
                         ::testing::ConvertGenerator<scales_param_tuple>(
                             ::testing::Combine(oop_packed_layout, bwd_only, interleaved_storage, ::testing::Values(3),
                                                ::testing::Values(sizes_t{9}, sizes_t{16}, sizes_t{64}, sizes_t{512},
                                                                  sizes_t{4096}, sizes_t{16, 512}),
                                                ::testing::Values(1.0), scales)),
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

#define INSTANTIATE_TESTS(TYPE)     \
  INSTANTIATE_TESTS_FULL(TYPE, usm) \
  //INSTANTIATE_TESTS_FULL(TYPE, buffer)

#endif
