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

#include <complex>
#include <gtest/gtest.h>

constexpr int N = 8;
constexpr int wg_size = 64;
using ftype = float;

class test_transfers_kernel_padded;
class test_transfers_kernel_unpadded;

TEST(transfers, unpadded) {
  std::vector<ftype> a, b;
  a.resize(N * wg_size);
  b.resize(N * wg_size);

  populate_with_random(a, ftype(-1.0), ftype(1.0));

  sycl::queue q;
  ftype* a_dev = sycl::malloc_device<ftype>(N * wg_size, q);
  ftype* b_dev = sycl::malloc_device<ftype>(N * wg_size, q);
  q.copy(a.data(), a_dev, N * wg_size);
  q.copy(b.data(), b_dev, N * wg_size);
  q.wait();

  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<ftype, 1> loc1(N * wg_size, h);
    sycl::local_accessor<ftype, 1> loc2(N * wg_size, h);
    h.parallel_for<test_transfers_kernel_unpadded>(sycl::nd_range<1>({wg_size}, {wg_size}), [=](sycl::nd_item<1> it) {
      size_t local_id = it.get_group().get_local_linear_id();

      ftype priv[N];

      sycl_fft::global2local<false>(a_dev, loc1, N * wg_size, wg_size, local_id);
      group_barrier(it.get_group());
      sycl_fft::local2private<N, false>(loc1, priv, local_id, N);
      sycl_fft::private2local<N, false>(priv, loc2, local_id, N);
      group_barrier(it.get_group());
      sycl_fft::local2global<false>(loc2, b_dev, N * wg_size, wg_size, local_id);
    });
  });

  q.wait();

  q.copy(b_dev, b.data(), N * wg_size);
  q.wait();

  compare_arrays(a, b, 0.0);
  sycl::free(a_dev, q);
  sycl::free(b_dev, q);
}

TEST(transfers, padded) {
  std::vector<ftype> a, b;
  a.resize(N * wg_size);
  b.resize(N * wg_size);

  populate_with_random(a, ftype(-1.0), ftype(1.0));

  sycl::queue q;
  ftype* a_dev = sycl::malloc_device<ftype>(N * wg_size, q);
  ftype* b_dev = sycl::malloc_device<ftype>(N * wg_size, q);
  q.copy(a.data(), a_dev, N * wg_size);
  q.copy(b.data(), b_dev, N * wg_size);
  q.wait();

  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<ftype, 1> loc1(sycl_fft::detail::pad_local(N * wg_size), h);
    sycl::local_accessor<ftype, 1> loc2(sycl_fft::detail::pad_local(N * wg_size), h);
    h.parallel_for<test_transfers_kernel_padded>(sycl::nd_range<1>({wg_size}, {wg_size}), [=](sycl::nd_item<1> it) {
      size_t local_id = it.get_group().get_local_linear_id();

      ftype priv[N];

      sycl_fft::global2local<true>(a_dev, loc1, N * wg_size, wg_size, local_id);
      group_barrier(it.get_group());
      sycl_fft::local2private<N, true>(loc1, priv, local_id, N);
      sycl_fft::private2local<N, true>(priv, loc2, local_id, N);
      group_barrier(it.get_group());
      sycl_fft::local2global<true>(loc2, b_dev, N * wg_size, wg_size, local_id);
    });
  });

  q.wait();

  q.copy(b_dev, b.data(), N * wg_size);
  q.wait();

  compare_arrays(a, b, 0.0);
  sycl::free(a_dev, q);
  sycl::free(b_dev, q);
}
