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

#ifndef SYCL_FFT_BENCH_LAUNCH_BENCH_HPP
#define SYCL_FFT_BENCH_LAUNCH_BENCH_HPP

#include <benchmark/benchmark.h>
#include <cassert>
#include <descriptor.hpp>
#include <traits.hpp>

#include "bench_utils.hpp"
#include "device_number_generator.hpp"
#include "enums.hpp"
#include "ops_estimate.hpp"
#include "reference_dft.hpp"

template <sycl_fft::direction dir, typename T>
void verify_dft(T* device_data, T* input, std::size_t batch, std::size_t N, sycl_fft::placement Placement,
                double scaling_factor = 1.0) {
  std::vector<T> host_result(N * batch);
  for (std::size_t i = 0; i < batch; i++) {
    const auto offset = i * N;
    reference_dft<dir>(input + offset, host_result.data() + offset, {static_cast<int>(N)}, scaling_factor);
  }
  bool correct = compare_arrays(device_data, host_result.data(), batch * N, 1e-5);
  if (!correct) {
    throw std::runtime_error("Verification Failed");
  }
}

template <typename ftype, sycl_fft::domain domain>
void bench_dft_real_time(benchmark::State& state, sycl_fft::descriptor<ftype, domain> desc) {
  using complex_type = std::complex<ftype>;
  std::size_t N = desc.get_total_length();
  std::size_t N_transforms = desc.number_of_transforms;
  std::size_t num_elements = N * N_transforms;
  double ops = cooley_tukey_ops_estimate(N, N_transforms);

  sycl::queue q;
  complex_type* in_dev = sycl::malloc_device<complex_type>(num_elements, q);
  complex_type* out_dev =
      desc.placement == sycl_fft::placement::IN_PLACE ? nullptr : sycl::malloc_device<complex_type>(num_elements, q);

  auto committed = desc.commit(q);
  q.wait();

#ifdef SYCLFFT_VERIFY_BENCHMARK
  memFill(in_dev, q, num_elements);
  std::vector<complex_type> host_input(num_elements);
  q.copy(in_dev, host_input.data(), num_elements).wait();
#endif  // SYCLFFT_VERIFY_BENCHMARK

  // warmup
  auto event = desc.placement == sycl_fft::placement::IN_PLACE ? committed.compute_forward(in_dev)
                                                               : committed.compute_forward(in_dev, out_dev);
  event.wait();

#ifdef SYCLFFT_VERIFY_BENCHMARK
  std::vector<complex_type> host_output(num_elements);
  q.copy(desc.placement == sycl_fft::placement::IN_PLACE ? in_dev : out_dev, host_output.data(), num_elements).wait();
  verify_dft<sycl_fft::direction::FORWARD>(host_input.data(), host_output.data(), N_transforms, N, desc.placement,
                                           desc.forward_scale);
#endif  // SYCLFFT_VERIFY_BENCHMARK

  for (auto _ : state) {
    // we need to manually measure time, so as to have it available here for the
    // calculation of flops
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    if (desc.placement == sycl_fft::placement::IN_PLACE) {
      start = std::chrono::high_resolution_clock::now();
      committed.compute_forward(in_dev).wait();
      end = std::chrono::high_resolution_clock::now();
    } else {
      start = std::chrono::high_resolution_clock::now();
      committed.compute_forward(in_dev, out_dev).wait();
      end = std::chrono::high_resolution_clock::now();
    }
    double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    state.counters["flops"] = ops / elapsed_seconds;
    state.SetIterationTime(elapsed_seconds);
  }
  sycl::free(in_dev, q);
  sycl::free(out_dev, q);
}

template <typename ftype, sycl_fft::domain domain>
void bench_dft_device_time(benchmark::State& state, sycl_fft::descriptor<ftype, domain> desc) {
  using complex_type = std::complex<ftype>;
  std::size_t N = desc.get_total_length();
  std::size_t N_transforms = desc.number_of_transforms;
  std::size_t num_elements = N * N_transforms;
  double ops = cooley_tukey_ops_estimate(N, N_transforms);

  sycl::queue q({sycl::property::queue::enable_profiling()});
  complex_type* in_dev = sycl::malloc_device<complex_type>(num_elements, q);
  complex_type* out_dev =
      desc.placement == sycl_fft::placement::IN_PLACE ? nullptr : sycl::malloc_device<complex_type>(num_elements, q);

  auto committed = desc.commit(q);

  q.wait();

#ifdef SYCLFFT_VERIFY_BENCHMARK
  memFill(in_dev, q, num_elements);
  std::vector<complex_type> host_input(num_elements);
  q.copy(in_dev, host_input.data(), num_elements).wait();
#endif  // SYCLFFT_VERIFY_BENCHMARK

  auto compute = [&]() {
    return desc.placement == sycl_fft::placement::IN_PLACE ? committed.compute_forward(in_dev)
                                                           : committed.compute_forward(in_dev, out_dev);
  };
  // warmup
  compute().wait();
#ifdef SYCLFFT_VERIFY_BENCHMARK
  std::vector<complex_type> host_output(num_elements);
  q.copy(desc.placement == sycl_fft::placement::IN_PLACE ? in_dev : out_dev, host_output.data(), num_elements).wait();
  verify_dft<sycl_fft::direction::FORWARD>(host_input.data(), host_output.data(), N_transforms, N, desc.placement,
                                           desc.forward_scale);
#endif  // SYCLFFT_VERIFY_BENCHMARK

  for (auto _ : state) {
    sycl::event e = compute();
    e.wait();
    int64_t start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    int64_t end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    double elapsed_seconds = (end - start) / 1e9;
    state.counters["flops"] = ops / elapsed_seconds;
    state.SetIterationTime(elapsed_seconds);
  }
  sycl::free(in_dev, q);
  sycl::free(out_dev, q);
}

template <typename ftype, sycl_fft::domain domain>
sycl_fft::descriptor<ftype, domain> create_descriptor(benchmark::State& state) {
  std::size_t N = state.range(0);
  sycl_fft::descriptor<ftype, sycl_fft::domain::COMPLEX> desc{{N}};
  desc.number_of_transforms = state.range(1);
  return desc;
}

template <typename T>
void bench_dft_real_time(benchmark::State& state) {
  using ftype = typename sycl_fft::get_real<T>::type;
  constexpr sycl_fft::domain domain = sycl_fft::get_domain<T>::value;
  auto desc = create_descriptor<ftype, domain>(state);
  bench_dft_real_time<ftype, domain>(state, desc);
}

template <typename T>
void bench_dft_device_time(benchmark::State& state) {
  using ftype = typename sycl_fft::get_real<T>::type;
  constexpr sycl_fft::domain domain = sycl_fft::get_domain<T>::value;
  auto desc = create_descriptor<ftype, domain>(state);
  bench_dft_device_time<ftype, domain>(state, desc);
}

#endif  // SYCL_FFT_BENCH_LAUNCH_BENCH_HPP
