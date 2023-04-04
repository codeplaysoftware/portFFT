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

#include <benchmark/benchmark.h>
#include <common/transfers.hpp>
#include <common/workitem.hpp>
#include <enums.hpp>
#include <iostream>

constexpr int N = 36;
constexpr int sg_size = 16;
constexpr int n_sgs = 100;
constexpr int n_transforms = sg_size * n_sgs;
using ftype = float;
using complex_type = std::complex<ftype>;

void init(int size, complex_type* a) {
  for (int i = 0; i < size; i++) {
    a[i] = {static_cast<ftype>(i * 0.3),
            static_cast<ftype>(((N - i) % 11) * 0.7)};
  }
}

template <typename T>
void DFT_dispatcher(T priv, std::size_t size) {
  switch (size) {
#define IMPL(N)                            \
  case N:                                  \
    sycl_fft::wi_dft<N, 1, 1>(priv, priv); \
    break;
    IMPL(1)
    IMPL(2)
    IMPL(3)
    IMPL(4)
    IMPL(5)
    IMPL(6)
    IMPL(7)
    IMPL(8)
    IMPL(9)
    IMPL(10)
    IMPL(11)
    IMPL(12)
    IMPL(13)
    IMPL(14)
    IMPL(15)
    IMPL(16)
    IMPL(17)
    IMPL(18)
    IMPL(19)
    IMPL(20)
    IMPL(21)
    IMPL(22)
    IMPL(23)
    IMPL(24)
    IMPL(25)
    IMPL(26)
    IMPL(27)
    IMPL(28)
    IMPL(29)
    IMPL(30)
    IMPL(31)
    IMPL(32)
    /*IMPL(33)
    IMPL(34)
    IMPL(35)
    IMPL(36)
    IMPL(37)
    IMPL(38)
    IMPL(39)
    IMPL(40)
    IMPL(41)
    IMPL(42)
    IMPL(43)
    IMPL(44)
    IMPL(45)
    IMPL(46)
    IMPL(47)
    IMPL(48)
    IMPL(49)
    IMPL(50)
    IMPL(51)
    IMPL(52)
    IMPL(53)
    IMPL(54)
    IMPL(55)
    IMPL(56)
    IMPL(57)
    IMPL(58)
    IMPL(59)
    IMPL(60)
    IMPL(61)
    IMPL(62)
    IMPL(63)
    IMPL(64)*/
  }
}

template <int N, typename T2_ptr>
void __attribute__((noinline)) dft_wrapper(T2_ptr in_out) {
  sycl_fft::wi_dft<sycl_fft::direction::FORWARD, N, 1, 1>(in_out, in_out);
}

constexpr static sycl::specialization_id<int> size_spec_const;

static void BM_dft(benchmark::State& state) {
  complex_type a[N * sg_size * n_sgs];
  complex_type b[N * sg_size * n_sgs];
  init(N * sg_size * n_sgs, a);

  sycl::queue q;
  complex_type* a_dev =
      sycl::malloc_device<complex_type>(N * sg_size * n_sgs, q);
  complex_type* b_dev =
      sycl::malloc_device<complex_type>(N * sg_size * n_sgs, q);
  q.copy(a, a_dev, N * sg_size * n_sgs);

  q.wait();

  auto run = [&]() {
    q.submit([&](sycl::handler& h) {
       h.set_specialization_constant<size_spec_const>(N);
       sycl::local_accessor<complex_type, 1> loc(N * sg_size, h);
       h.parallel_for(
           sycl::nd_range<1>({sg_size * n_sgs}, {sg_size}),
           [=](sycl::nd_item<1> it, sycl::kernel_handler kh) [
               [intel::reqd_sub_group_size(sg_size)]] {
             int Nn = kh.get_specialization_constant<size_spec_const>();
             sycl::sub_group sg = it.get_sub_group();
             size_t local_id = sg.get_local_linear_id();

             complex_type priv[N];

             sycl_fft::global2local(a_dev, loc, N * sg_size, sg_size, local_id);
             sycl::group_barrier(sg);
             sycl_fft::local2private<N>(loc, priv, local_id, sg_size);
             for (long j = 0; j < 10; j++) {
               dft_wrapper<N>(reinterpret_cast<ftype*>(priv));
             }
             sycl_fft::private2local<N>(priv, loc, local_id, sg_size);
             sycl::group_barrier(sg);
             sycl_fft::local2global(loc, b_dev, N * sg_size, sg_size, local_id);
           });
     }).wait();
  };

  // warmup
  run();

  for (auto _ : state) {
    run();
  }
  sycl::free(a_dev, q);
  sycl::free(b_dev, q);
}

BENCHMARK(BM_dft);

BENCHMARK_MAIN();
