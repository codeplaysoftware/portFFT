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
#include <iostream>

#include "ops_estimate.hpp"

using ftype = float;
using complex_type = std::complex<ftype>;
constexpr int N = 16;
constexpr int sg_size = 32;
constexpr int n_transforms = 1024*1024*1024 / sizeof(complex_type) / N;
constexpr int n_sgs = n_transforms / sg_size;

constexpr bool avoid_bank_conflicts = true;

void init(int size, complex_type* a) {
  for (int i = 0; i < size; i++) {
    a[i] = {static_cast<ftype>(i * 0.3),
            static_cast<ftype>(((N - i) % 11) * 0.7)};
  }
}

template <typename T>
void __attribute__((noinline)) DFT_dispatcher(T priv, std::size_t size) {
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
    /*IMPL(17)
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
    IMPL(33)
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
  sycl_fft::wi_dft<N, 1, 1>(in_out, in_out);
}

constexpr static sycl::specialization_id<int> size_spec_const;

static void BM_dft(benchmark::State& state) {
  double ops = cooley_tukey_ops_estimate(N, n_transforms);
  //complex_type a[N * n_transforms];
  //complex_type b[N * n_transforms];
  //init(N * n_transforms, a);

  sycl::queue q;
  size_t compute_units = q.get_device().get_info<sycl::info::device::max_compute_units>();
  ftype* a_dev =
      sycl::malloc_device<ftype>(2 * N * n_transforms, q);
  ftype* b_dev =
      sycl::malloc_device<ftype>(2 * N * n_transforms, q);
  //q.copy(a, a_dev, N * n_transforms);

  q.wait();

  auto run = [&]() {
    auto start = std::chrono::high_resolution_clock::now();
    q.submit([&](sycl::handler& h) {
       //h.set_specialization_constant<size_spec_const>(N);
       sycl::local_accessor<ftype, 1> loc(2 * (N+1) * sg_size, h);
       h.parallel_for(
           sycl::nd_range<1>({n_transforms/*sg_size * 8 * 64 * compute_units*/}, {/*4**/sg_size}),
           [=](sycl::nd_item<1> it, sycl::kernel_handler kh) [
               [intel::reqd_sub_group_size(sg_size)]] {
            constexpr int N_reals = 2*N;
             //int Nn = kh.get_specialization_constant<size_spec_const>();
             sycl::sub_group sg = it.get_sub_group();
             size_t local_id = sg.get_local_linear_id();
             size_t global_id = it.get_global_id(0);
             size_t global_size = it.get_global_range(0);

             ftype priv[N_reals];
             size_t i=global_id;
             //for(int i=global_id; i<n_transforms; i+=global_size){
              sycl_fft::global2local<avoid_bank_conflicts>(a_dev, loc, N_reals * sg_size, sg_size, local_id, 
                                      N_reals * (i - local_id));
              /*for (std::size_t k = local_id; k < N_reals * sg_size; k += sg_size*32) {
                for(std::size_t j=0; j< sg_size*32; j+=sg_size){
                  loc[0 + k + j + k/32] = a_dev[N_reals * (i - local_id) + k + j];
                }
              }*/
              /*for (std::size_t k = local_id; k < N_reals * sg_size; k += sg_size) {
                std::size_t local_idx = 0 + k;
                local_idx += local_idx/32;
                loc[local_idx] = a_dev[N_reals * (i - local_id) + k];
              }*/
              sycl::group_barrier(sg);
              sycl_fft::local2private<N_reals, avoid_bank_conflicts>(loc, priv, local_id, N_reals);
              /*for (std::size_t i = 0; i < N_reals; i++) {
                priv[i] = loc[0 + local_id * (N_reals+1) + i];
              }*/

              //dft_wrapper<N>(priv);
              sycl_fft::wi_dft<N, 1, 1>(priv, priv);
              //DFT_dispatcher(priv, N);

              sycl_fft::private2local<N_reals, avoid_bank_conflicts>(priv, loc, local_id, N_reals);
              /*for (std::size_t i = 0; i < N_reals; i++) {
                loc[0 + local_id * (N_reals+1) + i] = priv[i];
              }*/
              //loc[local_id] = priv[0];
              sycl::group_barrier(sg);
              sycl_fft::local2global<avoid_bank_conflicts>(loc, b_dev, N_reals * sg_size, sg_size, local_id, 
                                     0, N_reals * (i - local_id));
              /*for (std::size_t k = local_id; k < N_reals * sg_size; k += sg_size*32) {
                for(std::size_t j=0; j< sg_size*32; j+=sg_size){
                  a_dev[N_reals * (i - local_id) + k + j] = loc[0 + k + j + k/32];
                }
              }*/
              /*for (std::size_t k = local_id; k < N_reals * sg_size; k += sg_size) {
                std::size_t local_idx = 0 + k;
                local_idx += local_idx/32;
                b_dev[N_reals * (i - local_id) + k] = loc[local_idx];
              }*/
             //}
           });
     }).wait();
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
    state.counters["flops"] = ops / elapsed_seconds;
    state.SetIterationTime(elapsed_seconds);
  };


  // warmup
  run();

  for (auto _ : state) {
    run();
  }
  sycl::free(a_dev, q);
  sycl::free(b_dev, q);
}

BENCHMARK(BM_dft)->UseManualTime();

BENCHMARK_MAIN();
