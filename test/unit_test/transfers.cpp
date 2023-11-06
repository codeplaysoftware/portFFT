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

#include <gtest/gtest.h>
#include <portfft/common/memory_views.hpp>
#include <portfft/common/transfers.hpp>

#include "fft_test_utils.hpp"

constexpr int N = 4;
constexpr int sg_size = (PORTFFT_SUBGROUP_SIZES);  // turn the list into the last value using commma operator
constexpr int wg_size = sg_size * PORTFFT_SGS_IN_WG;
constexpr int N_sentinel_values = 64;
using ftype = float;
constexpr ftype sentinel_a = -999;
constexpr ftype sentinel_b = -888;
constexpr ftype sentinel_loc1 = -777;
constexpr ftype sentinel_loc2 = -666;

template <portfft::detail::pad Pad, std::size_t BankGroupsPerPad>
class test_transfers_kernel;

template <portfft::detail::pad Pad, std::size_t BankGroupsPerPad>
void test() {
  std::vector<ftype> a, b;
  a.resize(N * wg_size);
  b.resize(N * wg_size);

  for (std::size_t i = 0; i < N * wg_size; i++) {
    a[i] = static_cast<ftype>(i);
  }

  sycl::queue q;
  ftype* sentinels_loc1_dev = sycl::malloc_device<ftype>(2 * N_sentinel_values, q);
  ftype* sentinels_loc2_dev = sycl::malloc_device<ftype>(2 * N_sentinel_values, q);

  ftype* a_dev = sycl::malloc_device<ftype>(N * wg_size + 2 * N_sentinel_values, q);
  ftype* b_dev = sycl::malloc_device<ftype>(N * wg_size + 2 * N_sentinel_values, q);
  ftype* a_dev_work = a_dev + N_sentinel_values;
  ftype* b_dev_work = b_dev + N_sentinel_values;

  q.fill(a_dev, sentinel_a, N_sentinel_values);
  q.fill(a_dev + N * wg_size + N_sentinel_values, sentinel_a, N_sentinel_values);
  q.copy(a.data(), a_dev_work, N * wg_size);
  q.fill(b_dev, sentinel_b, N * wg_size + 2 * N_sentinel_values);
  q.wait();

  std::size_t padded_local_size =
      static_cast<std::size_t>(portfft::detail::pad_local<Pad>(N * wg_size, static_cast<int>(BankGroupsPerPad)));

  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<ftype, 1> loc1(padded_local_size + 2 * N_sentinel_values, h);
    sycl::local_accessor<ftype, 1> loc2(padded_local_size + 2 * N_sentinel_values, h);
#ifdef PORTFFT_LOG
    sycl::stream s{1024 * 8, 1024, h};
#endif
    h.parallel_for<test_transfers_kernel<Pad, BankGroupsPerPad>>(
        sycl::nd_range<1>({wg_size}, {wg_size}), [=](sycl::nd_item<1> it) {
          detail::global_data_struct global_data{
#ifdef PORTFFT_LOG
              s,
#endif
              it};
          portfft::Idx local_id = static_cast<portfft::Idx>(it.get_group().get_local_linear_id());

          ftype priv[N];
          ftype* loc1_work = &loc1[N_sentinel_values];
          ftype* loc2_work = &loc2[N_sentinel_values];
          if (local_id == 0) {
            for (std::size_t i = 0; i < padded_local_size + 2 * N_sentinel_values; i++) {
              loc1[i] = sentinel_loc1;
              loc2[i] = sentinel_loc2;
            }
          }
          auto loc1_view = detail::padded_view(loc1_work, BankGroupsPerPad);
          auto loc2_view = detail::padded_view(loc2_work, BankGroupsPerPad);
          group_barrier(it.get_group());
          portfft::global2local<detail::level::WORKGROUP, sg_size>(global_data, a_dev_work, loc1_view, N * wg_size);
          group_barrier(it.get_group());
          copy_wi(global_data, detail::offset_view{loc1_view, local_id * N}, priv, N);
          copy_wi(global_data, priv, detail::offset_view{loc2_view, local_id * N}, N);
          group_barrier(it.get_group());
          portfft::local2global<detail::level::WORKGROUP, sg_size>(global_data, loc2_view, b_dev_work, N * wg_size);
          group_barrier(it.get_group());
          if (local_id == 0) {
            for (std::size_t i = 0; i < N_sentinel_values; i++) {
              sentinels_loc1_dev[i] = loc1[i];
              sentinels_loc2_dev[i] = loc2[i];
            }
            for (std::size_t i = 0; i < N_sentinel_values; i++) {
              sentinels_loc1_dev[N_sentinel_values + i] = loc1[padded_local_size + N_sentinel_values + i];
              sentinels_loc2_dev[N_sentinel_values + i] = loc2[padded_local_size + N_sentinel_values + i];
            }
          }
        });
  });

  q.wait();

  std::vector<ftype> b_sentinels_start(N_sentinel_values);
  std::vector<ftype> b_sentinels_end(N_sentinel_values);
  std::vector<ftype> loc1_sentinels(N_sentinel_values * 2);
  std::vector<ftype> loc2_sentinels(N_sentinel_values * 2);
  q.copy(sentinels_loc1_dev, loc1_sentinels.data(), N_sentinel_values * 2);
  q.copy(sentinels_loc2_dev, loc2_sentinels.data(), N_sentinel_values * 2);
  q.copy(b_dev, b_sentinels_start.data(), N_sentinel_values);
  q.copy(b_dev + N * wg_size + N_sentinel_values, b_sentinels_end.data(), N_sentinel_values);
  q.copy(b_dev_work, b.data(), N * wg_size);
  q.wait();

  expect_arrays_eq(a, b);
  for (std::size_t i = 0; i < N_sentinel_values; i++) {
    EXPECT_EQ(b_sentinels_start[i], sentinel_b);
    EXPECT_EQ(b_sentinels_end[i], sentinel_b);
  }
  for (std::size_t i = 0; i < N_sentinel_values * 2; i++) {
    EXPECT_EQ(loc1_sentinels[i], sentinel_loc1);
    EXPECT_EQ(loc2_sentinels[i], sentinel_loc2);
  }
  sycl::free(a_dev, q);
  sycl::free(b_dev, q);
  sycl::free(sentinels_loc1_dev, q);
  sycl::free(sentinels_loc2_dev, q);
}

TEST(transfers, unpadded) { test<portfft::detail::pad::DONT_PAD, 0>(); }

TEST(transfers, padded1) { test<portfft::detail::pad::DO_PAD, 1>(); }
TEST(transfers, padded3) { test<portfft::detail::pad::DO_PAD, 3>(); }
TEST(transfers, padded4) { test<portfft::detail::pad::DO_PAD, 4>(); }
