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

#include <cassert>
#include <sstream>
#include <type_traits>

#include <benchmark/benchmark.h>

#include <descriptor.hpp>
#include <enums.hpp>

#include "bench_utils.hpp"
#include "device_number_generator.hpp"
#include "ops_estimate.hpp"

/**
 * Main function to run benchmarks and measure the time spent on the host.
 * One GBench iteration consists of multiple compute submitted asynchronously to reduce the overhead of the SYCL
 * runtime. The function is used in \p bench_float and \p bench_manual_(float|double) . The function throws exception if
 * an error occurs.
 *
 * @tparam ftype float or double
 * @tparam domain COMPLEX or REAL
 * @param state GBench state
 * @param q Queue to use
 * @param desc Description of the FFT problem
 * @param runs Number of asynchronous compute in one GBench iteration
 */
template <typename ftype, sycl_fft::domain domain>
void bench_dft_average_host_time_impl(benchmark::State& state, sycl::queue q, sycl_fft::descriptor<ftype, domain> desc,
                                      std::size_t runs) {
  using complex_type = std::complex<ftype>;
  std::size_t N = desc.get_total_length();
  std::size_t N_transforms = desc.number_of_transforms;
  std::size_t num_elements = N * N_transforms;
  double ops = cooley_tukey_ops_estimate(N, N_transforms);
  std::size_t bytes_transferred = global_mem_transactions<complex_type, complex_type>(N_transforms, N, N);

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
  verify_dft(host_input.data(), host_output.data(), std::vector<std::size_t>{N}, N_transforms, desc.forward_scale);
#endif  // SYCLFFT_VERIFY_BENCHMARK
  std::vector<sycl::event> dependencies;
  dependencies.reserve(1);

  for (auto _ : state) {
    // we need to manually measure time, so as to have it available here for the
    // calculation of flops
    dependencies.clear();

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    if (desc.placement == sycl_fft::placement::IN_PLACE) {
      start = std::chrono::high_resolution_clock::now();
      dependencies.emplace_back(committed.compute_forward(in_dev));
      for (std::size_t r = 1; r != runs; r += 1) {
        dependencies[0] = committed.compute_forward(in_dev, dependencies);
      }
      dependencies[0].wait();
      end = std::chrono::high_resolution_clock::now();
    } else {
      start = std::chrono::high_resolution_clock::now();
      dependencies.emplace_back(committed.compute_forward(in_dev, out_dev));
      for (std::size_t r = 1; r != runs; r += 1) {
        dependencies[0] = committed.compute_forward(in_dev, out_dev, dependencies);
      }
      dependencies[0].wait();
      end = std::chrono::high_resolution_clock::now();
    }
    double elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() / static_cast<double>(runs);
    state.counters["flops"] = ops / elapsed_seconds;
    state.counters["throughput"] = static_cast<double>(bytes_transferred) / elapsed_seconds;
    state.SetIterationTime(elapsed_seconds);
  }
  sycl::free(in_dev, q);
  sycl::free(out_dev, q);
}

/**
 * Separate impl function to handle catching exceptions
 * @see bench_dft_average_host_time_impl
 */
template <typename ftype, sycl_fft::domain domain>
void bench_dft_average_host_time(benchmark::State& state, sycl::queue q, sycl_fft::descriptor<ftype, domain> desc) {
  try {
    bench_dft_average_host_time_impl(state, q, desc, runs_to_average);
  } catch (std::exception& e) {
    handle_exception(state, e);
  }
}

/**
 * Main function to run benchmarks and measure the time spent on the device.
 * The function is used in \p bench_float and \p bench_manual_(float|double) .
 * The function throws exception if an error occurs.
 *
 * @tparam ftype float or double
 * @tparam domain COMPLEX or REAL
 * @param state GBench state
 * @param q Queue to use, \p enable_profiling property must be set
 * @param desc Description of the FFT problem
 */
template <typename ftype, sycl_fft::domain domain>
void bench_dft_device_time_impl(benchmark::State& state, sycl::queue q, sycl_fft::descriptor<ftype, domain> desc) {
  using complex_type = std::complex<ftype>;
  if (!q.has_property<sycl::property::queue::enable_profiling>()) {
    throw std::runtime_error("Queue does not have the profiling property");
  }

  std::size_t N = desc.get_total_length();
  std::size_t N_transforms = desc.number_of_transforms;
  std::size_t num_elements = N * N_transforms;
  double ops = cooley_tukey_ops_estimate(N, N_transforms);
  std::size_t bytes_transferred = global_mem_transactions<complex_type, complex_type>(N_transforms, N, N);

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
  verify_dft(host_input.data(), host_output.data(), std::vector<std::size_t>{N}, N_transforms, desc.forward_scale);
#endif  // SYCLFFT_VERIFY_BENCHMARK

  for (auto _ : state) {
    sycl::event e = compute();
    e.wait();
    auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    double elapsed_seconds = static_cast<double>(end - start) / 1e9;
    state.counters["flops"] = ops / elapsed_seconds;
    state.counters["throughput"] = static_cast<double>(bytes_transferred) / elapsed_seconds;
    state.SetIterationTime(elapsed_seconds);
  }
  sycl::free(in_dev, q);
  sycl::free(out_dev, q);
}

/**
 * Separate impl function to handle catching exceptions
 * @see bench_dft_device_time_impl
 */
template <typename ftype, sycl_fft::domain domain>
void bench_dft_device_time(benchmark::State& state, sycl::queue q, sycl_fft::descriptor<ftype, domain> desc) {
  try {
    bench_dft_device_time_impl(state, q, desc);
  } catch (std::exception& e) {
    handle_exception(state, e);
  }
}

/**
 * Helper function to register each benchmark configuration twice, once for measuring the time on host and once
 * for measuring on device.
 *
 * @tparam ftype float or double
 * @tparam domain COMPLEX or REAL
 * @param suffix Suffix for the benchmark name
 * @param q Queue used for profiling the time on the host
 * @param profiling_q Queue used for profiling the time on the device
 * @param desc Description of the FFT problem
 */
template <typename ftype, sycl_fft::domain domain>
void register_host_device_benchmark(const std::string& suffix, sycl::queue q, sycl::queue profiling_q,
                                    const sycl_fft::descriptor<ftype, domain>& desc) {
  static_assert(domain == sycl_fft::domain::REAL || domain == sycl_fft::domain::COMPLEX, "Unsupported domain");
  static_assert(std::is_same<ftype, float>::value || std::is_same<ftype, double>::value, "Unsupported precision");
  // Print descriptor's parameters relevant for benchmarks
  // Additional parameters could be added to the suffix if needed
  auto print_desc = [&](std::ostream& name) {
    name << "d=" << (domain == sycl_fft::domain::REAL ? "re" : "cpx");
    name << ",prec=" << (std::is_same<ftype, float>::value ? "single" : "double");
    name << ",n=[";
    for (std::size_t i = 0; i < desc.lengths.size(); ++i) {
      name << (i > 0 ? ", " : "") << desc.lengths[i];
    }
    name << "],batch=" << desc.number_of_transforms;
  };

  std::stringstream bench_host_name;
  bench_host_name << "average_host_time/";
  print_desc(bench_host_name);
  bench_host_name << "/" << suffix;
  benchmark::RegisterBenchmark(bench_host_name.str().c_str(), bench_dft_average_host_time<ftype, domain>, q, desc)
      ->UseManualTime();

  std::stringstream bench_device_name;
  bench_device_name << "device_time/";
  print_desc(bench_device_name);
  bench_device_name << "/" << suffix;
  benchmark::RegisterBenchmark(bench_device_name.str().c_str(), bench_dft_device_time<ftype, domain>, profiling_q, desc)
      ->UseManualTime();
}

#endif  // SYCL_FFT_BENCH_LAUNCH_BENCH_HPP
