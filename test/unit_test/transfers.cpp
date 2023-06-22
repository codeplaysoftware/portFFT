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

#include "common/transfers.hpp"
#include "number_generators.hpp"
#include "utils.hpp"

#include <gtest/gtest.h>

constexpr int N = 4;
constexpr int sg_size = (SYCLFFT_SUBGROUP_SIZES);  // turn the list into the last value using commma operator
constexpr int wg_size = sg_size * SYCLFFT_SGS_IN_WG;
constexpr int N_sentinel_values = 64;
using ftype = float;
constexpr ftype sentinel_a = -999;
constexpr ftype sentinel_b = -888;
constexpr ftype sentinel_loc1 = -777;
constexpr ftype sentinel_loc2 = -666;

template <sycl_fft::detail::pad Pad>
class test_transfers_kernel;

template <sycl_fft::detail::pad Pad>
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

  std::size_t padded_local_size = sycl_fft::detail::pad_local<Pad>(N * wg_size);

  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<ftype, 1> loc1(padded_local_size + 2 * N_sentinel_values, h);
    sycl::local_accessor<ftype, 1> loc2(padded_local_size + 2 * N_sentinel_values, h);
    h.parallel_for<test_transfers_kernel<Pad>>(sycl::nd_range<1>({wg_size}, {wg_size}), [=](sycl::nd_item<1> it) {
      std::size_t local_id = it.get_group().get_local_linear_id();

      ftype priv[N];
      ftype* loc1_work = &loc1[N_sentinel_values];
      ftype* loc2_work = &loc2[N_sentinel_values];
      if (local_id == 0) {
        for (std::size_t i = 0; i < padded_local_size + 2 * N_sentinel_values; i++) {
          loc1[i] = sentinel_loc1;
          loc2[i] = sentinel_loc2;
        }
      }
      group_barrier(it.get_group());
      sycl_fft::global2local<Pad, detail::level::WORKGROUP, sg_size>(it, a_dev_work, loc1_work, N * wg_size);
      group_barrier(it.get_group());
      sycl_fft::local2private<N, Pad>(loc1_work, priv, local_id, N);
      sycl_fft::private2local<N, Pad>(priv, loc2_work, local_id, N);
      group_barrier(it.get_group());
      sycl_fft::local2global<Pad, detail::level::WORKGROUP, sg_size>(it, loc2_work, b_dev_work, N * wg_size);
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

  compare_arrays(a, b, 0.0);
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

TEST(transfers, unpadded) { test<sycl_fft::detail::pad::DONT_PAD>(); }

TEST(transfers, padded) { test<sycl_fft::detail::pad::DO_PAD>(); }
