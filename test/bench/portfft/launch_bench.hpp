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

#ifndef PORTFFT_BENCH_LAUNCH_BENCH_HPP
#define PORTFFT_BENCH_LAUNCH_BENCH_HPP

#include <cassert>
#include <sstream>
#include <type_traits>

#include <benchmark/benchmark.h>
#include <portfft/portfft.hpp>

#include "utils/bench_utils.hpp"
#include "utils/device_number_generator.hpp"
#include "utils/ops_estimate.hpp"

/**
 * Main function to run benchmarks and measure the time spent on the host.
 * One GBench iteration consists of multiple compute submitted asynchronously to reduce the overhead of the SYCL
 * runtime. The function is used in \p bench_float and \p bench_manual_(float|double) . The function throws exception if
 * an error occurs.
 *
 * @tparam FType float or double
 * @tparam Domain COMPLEX or REAL
 * @param state GBench state
 * @param q Queue to use
 * @param desc Description of the FFT problem
 * @param runs Number of asynchronous compute in one GBench iteration
 */
template <typename FType, portfft::domain Domain>
void bench_dft_average_host_time_impl(benchmark::State& state, sycl::queue q, portfft::descriptor<FType, Domain> desc,
                                      std::size_t runs) {
  using complex_type = std::complex<FType>;
  using forward_t = std::conditional_t<Domain == portfft::domain::COMPLEX, complex_type, FType>;
  std::size_t N = desc.get_flattened_length();
  std::size_t N_transforms = desc.number_of_transforms;
  std::size_t num_elements = N * N_transforms;
  double ops = cooley_tukey_ops_estimate(N, N_transforms);
  std::size_t bytes_transferred = global_mem_transactions<complex_type, complex_type>(N_transforms, N, N);

  forward_t* in_dev = sycl::malloc_device<forward_t>(num_elements, q);
  complex_type* out_dev =
      desc.placement == portfft::placement::IN_PLACE ? nullptr : sycl::malloc_device<complex_type>(num_elements, q);

  auto committed = desc.commit(q);
  q.wait();

#ifdef PORTFFT_VERIFY_BENCHMARKS
  auto [forward_data, backward_data] = gen_fourier_data<portfft::direction::FORWARD>(
      desc, portfft::detail::layout::PACKED, portfft::detail::layout::PACKED);
  q.copy(forward_data.data(), in_dev, num_elements).wait();
#endif  // PORTFFT_VERIFY_BENCHMARKS

  // warmup
  auto event = desc.placement == portfft::placement::IN_PLACE ? committed.compute_forward(in_dev)
                                                              : committed.compute_forward(in_dev, out_dev);
  event.wait();

#ifdef PORTFFT_VERIFY_BENCHMARKS
  std::vector<complex_type> host_output(num_elements);
  q.copy(desc.placement == portfft::placement::IN_PLACE ? reinterpret_cast<complex_type*>(in_dev) : out_dev,
         host_output.data(), num_elements)
      .wait();
  verify_dft<portfft::direction::FORWARD>(desc, backward_data, host_output, 1e-2);
#endif  // PORTFFT_VERIFY_BENCHMARKS
  std::vector<sycl::event> dependencies;
  dependencies.reserve(1);

  for (auto _ : state) {
    // we need to manually measure time, so as to have it available here for the
    // calculation of flops
    dependencies.clear();

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    if (desc.placement == portfft::placement::IN_PLACE) {
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
template <typename FType, portfft::domain Domain>
void bench_dft_average_host_time(benchmark::State& state, sycl::queue q, portfft::descriptor<FType, Domain> desc) {
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
 * @tparam FType float or double
 * @tparam Domain COMPLEX or REAL
 * @param state GBench state
 * @param q Queue to use, \p enable_profiling property must be set
 * @param desc Description of the FFT problem
 */
template <typename FType, portfft::domain Domain>
void bench_dft_device_time_impl(benchmark::State& state, sycl::queue q, portfft::descriptor<FType, Domain> desc) {
  using complex_type = std::complex<FType>;
  using forward_t = std::conditional_t<Domain == portfft::domain::COMPLEX, complex_type, FType>;
  if (!q.has_property<sycl::property::queue::enable_profiling>()) {
    throw std::runtime_error("Queue does not have the profiling property");
  }

  std::size_t N = desc.get_flattened_length();
  std::size_t N_transforms = desc.number_of_transforms;
  std::size_t num_elements = N * N_transforms;
  double ops = cooley_tukey_ops_estimate(N, N_transforms);
  std::size_t bytes_transferred = global_mem_transactions<complex_type, complex_type>(N_transforms, N, N);

  forward_t* in_dev = sycl::malloc_device<forward_t>(num_elements, q);
  complex_type* out_dev =
      desc.placement == portfft::placement::IN_PLACE ? nullptr : sycl::malloc_device<complex_type>(num_elements, q);

  auto committed = desc.commit(q);

  q.wait();
#ifdef PORTFFT_VERIFY_BENCHMARKS
  auto [forward_data, backward_data] = gen_fourier_data<portfft::direction::FORWARD>(
      desc, portfft::detail::layout::PACKED, portfft::detail::layout::PACKED);
  q.copy(forward_data.data(), in_dev, num_elements).wait();
#endif  // PORTFFT_VERIFY_BENCHMARKS

  auto compute = [&]() {
    return desc.placement == portfft::placement::IN_PLACE ? committed.compute_forward(in_dev)
                                                          : committed.compute_forward(in_dev, out_dev);
  };
  // warmup
  compute().wait();

#ifdef PORTFFT_VERIFY_BENCHMARKS
  std::vector<complex_type> host_output(num_elements);
  q.copy(desc.placement == portfft::placement::IN_PLACE ? reinterpret_cast<complex_type*>(in_dev) : out_dev,
         host_output.data(), num_elements)
      .wait();
  verify_dft<portfft::direction::FORWARD>(desc, backward_data, host_output, 1e-2);
#endif  // PORTFFT_VERIFY_BENCHMARKS

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
template <typename FType, portfft::domain Domain>
void bench_dft_device_time(benchmark::State& state, sycl::queue q, portfft::descriptor<FType, Domain> desc) {
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
 * @tparam FType float or double
 * @tparam Domain COMPLEX or REAL
 * @param suffix Suffix for the benchmark name
 * @param q Queue used for profiling the time on the host
 * @param profiling_q Queue used for profiling the time on the device
 * @param desc Description of the FFT problem
 */
template <typename FType, portfft::domain Domain>
void register_host_device_benchmark(const std::string& suffix, sycl::queue q, sycl::queue profiling_q,
                                    const portfft::descriptor<FType, Domain>& desc) {
  static_assert(Domain == portfft::domain::REAL || Domain == portfft::domain::COMPLEX, "Unsupported domain");
  static_assert(std::is_same_v<FType, float> || std::is_same_v<FType, double>, "Unsupported precision");
  // Print descriptor's parameters relevant for benchmarks
  // Additional parameters could be added to the suffix if needed
  auto print_desc = [&](std::ostream& name) {
    name << "d=" << (Domain == portfft::domain::REAL ? "re" : "cpx");
    name << ",prec=" << (std::is_same_v<FType, float> ? "single" : "double");
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
  benchmark::RegisterBenchmark(bench_host_name.str().c_str(), bench_dft_average_host_time<FType, Domain>, q, desc)
      ->UseManualTime();

  std::stringstream bench_device_name;
  bench_device_name << "device_time/";
  print_desc(bench_device_name);
  bench_device_name << "/" << suffix;
  benchmark::RegisterBenchmark(bench_device_name.str().c_str(), bench_dft_device_time<FType, Domain>, profiling_q, desc)
      ->UseManualTime();
}

#endif  // PORTFFT_BENCH_LAUNCH_BENCH_HPP
