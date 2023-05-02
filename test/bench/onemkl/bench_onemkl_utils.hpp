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
 *  Common benchmark for close-source and open-source oneMKL libraries.
 *  The oneMKL headers must be included before this header.
 *
 **************************************************************************/

#include <complex>
#include <type_traits>
#include <vector>

#include <benchmark/benchmark.h>

#include "bench_utils.hpp"
#include "number_generators.hpp"
#include "ops_estimate.hpp"
#include "reference_dft_set.hpp"

/// Get the floating-point type from the MKL precision enum.
template <oneapi::mkl::dft::precision prec>
using get_float_t = std::conditional_t<prec == oneapi::mkl::dft::precision::SINGLE, float, double>;

/**
 * @brief Copy an input vector to an output vector, with element-wise casts.
 *
 * @tparam TOut output type
 * @tparam TIn input type
 * @param in_vec vector to cast
 * @return std::vector<TOut> the casted vector
 */
template <typename TOut, typename TIn>
std::vector<TOut> cast_vector_elements(const std::vector<TIn>& in_vec) {
  std::vector<TOut> out_vec(in_vec.size());
  for (int i{0}; i < in_vec.size(); ++i) {
    out_vec[i] = static_cast<TOut>(in_vec[i]);
  }
  return out_vec;
}

/** A class to own DFT descriptor and benchmark USM allocations. Currently
 * out-of-place DFT only.
 * @tparam prec DFT precision
 * @tparam domain DFT domain
 */
template <oneapi::mkl::dft::precision prec, oneapi::mkl::dft::domain domain>
struct onemkl_state {
  using descriptor_t = oneapi::mkl::dft::descriptor<prec, domain>;
  using float_t = get_float_t<prec>;
  using complex_t = std::complex<float_t>;

  // Constructor. Allocates required memory.
  onemkl_state(sycl::queue sycl_queue, std::vector<std::int64_t> lengths, std::int64_t number_of_transforms)
      : desc(lengths), sycl_queue(sycl_queue), lengths{lengths}, number_of_transforms{number_of_transforms} {
    using config_param_t = oneapi::mkl::dft::config_param;
    // For now, out-of-place only.
    set_out_of_place();
    desc.set_value(config_param_t::NUMBER_OF_TRANSFORMS, number_of_transforms);
    num_elements = get_total_length() * number_of_transforms;
    // Allocate memory.
    in_dev = sycl::malloc_device<complex_t>(num_elements, sycl_queue);
    out_dev = sycl::malloc_device<complex_t>(num_elements, sycl_queue);
  }

  ~onemkl_state() {
    sycl::free(in_dev, sycl_queue);
    sycl::free(out_dev, sycl_queue);
  }

  // Backend specific methods
  void set_out_of_place();
  inline sycl::event compute(const std::vector<sycl::event>& deps);

  /// The count of elements for each FFT. Product of lengths.
  inline std::size_t get_total_length() {
    return std::accumulate(lengths.cbegin(), lengths.cend(), 1, std::multiplies<std::int64_t>());
  }

  // Queue & allocations for test
  descriptor_t desc;
  sycl::queue sycl_queue;
  complex_t* in_dev = nullptr;
  complex_t* out_dev = nullptr;

  // Descriptor data to avoid having to use get_value.
  std::vector<std::int64_t> lengths;
  std::int64_t number_of_transforms;
  std::size_t num_elements;
};

/*** Benchmark a DFT on the host.
 * @tparam prec The DFT precision.
 * @tparam domain The DFT domain.
 * @param state Google benchmark state.
 * @param lengths The lengths defining and N-dimensional DFT.
 * @param number_of_transforms The DFT batch size.
 */
template <oneapi::mkl::dft::precision prec, oneapi::mkl::dft::domain domain>
void bench_dft_real_time(benchmark::State& state, std::vector<int> lengths, int number_of_transforms) {
  using float_type = get_float_t<prec>;
  using complex_type = std::complex<float_type>;
  sycl::queue q;
  auto lengthsI64 = cast_vector_elements<std::int64_t>(lengths);
  onemkl_state<prec, domain> fft_state{q, lengthsI64, number_of_transforms};
  std::size_t N = fft_state.get_total_length();
  double ops = cooley_tukey_ops_estimate(N, fft_state.number_of_transforms);
  std::size_t bytes_transfered =
      global_mem_transactions<complex_type, complex_type>(fft_state.number_of_transforms, N, N);

#ifdef SYCLFFT_VERIFY_BENCHMARK
  std::vector<complex_type> host_data(fft_state.num_elements);
  populate_with_random(host_data);

  q.copy(host_data.data(), fft_state.in_dev, fft_state.num_elements).wait_and_throw();
#endif

  const std::vector<sycl::event> no_dependencies;

  try {
    fft_state.desc.commit(q);
    q.wait_and_throw();
    // warmup
    fft_state.compute(no_dependencies).wait_and_throw();
  } catch (...) {
    // Can't run this benchmark!
    state.SkipWithError("Exception thrown: commit or warm-up failed.");
    return;
  }

  for (auto _ : state) {
    // we need to manually measure time, so as to have it available here for the
    // calculation of flops
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    fft_state.compute(no_dependencies).wait();
    auto end = clock::now();
    double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    state.counters["flops"] = ops / elapsed_seconds;
    state.counters["throughput"] = bytes_transfered / elapsed_seconds;
    state.SetIterationTime(elapsed_seconds);
  }
}

/*** Benchmark a DFT using device rather than host time.
 * @tparam prec The DFT precision.
 * @tparam domain The DFT domain.
 * @param state Google benchmark state.
 * @param lengths The lengths defining and N-dimensional DFT.
 * @param number_of_transforms The DFT batch size.
 */
template <oneapi::mkl::dft::precision prec, oneapi::mkl::dft::domain domain>
void bench_dft_device_time(benchmark::State& state, std::vector<int> lengths, int number_of_transforms) {
  using float_type = get_float_t<prec>;
  using complex_type = std::complex<float_type>;

  // Get key information out of the descriptor.
  sycl::queue q({sycl::property::queue::enable_profiling()});
  auto lengthsI64 = cast_vector_elements<std::int64_t>(lengths);
  onemkl_state<prec, domain> fft_state{q, lengthsI64, number_of_transforms};

  std::size_t N = fft_state.get_total_length();
  double ops = cooley_tukey_ops_estimate(N, fft_state.number_of_transforms);

  std::size_t bytes_transfered =
      global_mem_transactions<complex_type, complex_type>(fft_state.number_of_transforms, N, N);

#ifdef SYCLFFT_VERIFY_BENCHMARK
  std::vector<complex_type> host_data(fft_state.num_elements);
  populate_with_random(host_data);

  q.copy(host_data.data(), fft_state.in_dev, fft_state.num_elements).wait_and_throw();
#endif

  const std::vector<sycl::event> no_dependencies;

  try {
    fft_state.desc.commit(q);
    q.wait_and_throw();
    // warmup
    fft_state.compute(no_dependencies).wait_and_throw();
  } catch (...) {
    state.SkipWithError("Exception thrown: commit or warm-up failed.");
    return;
  }

  for (auto _ : state) {
    int64_t start{0}, end{0};
    try {
      sycl::event e = fft_state.compute(no_dependencies);
      e.wait();
      start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
      end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    } catch (sycl::exception& e) {
      // e may not have profiling info, so this benchmark is useless
      auto errorMessage = std::string("Exception thrown ") + e.what();
      state.SkipWithError(errorMessage.c_str());
      start = end;
    }
    double elapsed_seconds = (end - start) / 1e9;
    state.counters["flops"] = ops / elapsed_seconds;
    state.counters["throughput"] = bytes_transfered / elapsed_seconds;
    state.SetIterationTime(elapsed_seconds);
  }
}

/*** Benchmark average host time over many DFT runs to amortize error.
 * @tparam runs The number of DFT runs to average
 * @tparam prec The DFT precision.
 * @tparam domain The DFT domain.
 * @param state Google benchmark state.
 * @param lengths The lengths defining and N-dimensional DFT.
 * @param number_of_transforms The DFT batch size.
 */
template <std::size_t runs, oneapi::mkl::dft::precision prec, oneapi::mkl::dft::domain domain>
void bench_dft_average_host_time(benchmark::State& state, std::vector<int> lengths, int number_of_transforms) {
  using float_type = get_float_t<prec>;
  using complex_type = std::complex<float_type>;
  sycl::queue q;
  auto lengthsI64 = cast_vector_elements<std::int64_t>(lengths);
  onemkl_state<prec, domain> fft_state{q, lengthsI64, number_of_transforms};
  std::size_t N = fft_state.get_total_length();
  double ops = cooley_tukey_ops_estimate(N, fft_state.number_of_transforms);
  std::size_t bytes_transfered =
      global_mem_transactions<complex_type, complex_type>(fft_state.number_of_transforms, N, N);

#ifdef SYCLFFT_VERIFY_BENCHMARK
  std::vector<complex_type> host_data(fft_state.num_elements);
  populate_with_random(host_data);

  q.copy(host_data.data(), fft_state.in_dev, fft_state.num_elements).wait_and_throw();
#endif

  std::vector<sycl::event> dependencies;
  dependencies.reserve(1);

  try {
    fft_state.desc.commit(q);
    q.wait_and_throw();
    // warmup
    fft_state.compute(dependencies).wait_and_throw();
  } catch (...) {
    // Can't run this benchmark!
    state.SkipWithError("Exception thrown: commit or warm-up failed.");
    return;
  }

  for (auto _ : state) {
    dependencies.clear();
    // we need to manually measure time, so as to have it available here for the
    // calculation of flops
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    static_assert(runs >= 1);
    dependencies.emplace_back(fft_state.compute(dependencies));
    for (std::size_t r = 1; r != runs; r += 1) {
      dependencies[0] = fft_state.compute(dependencies);
    }
    dependencies[0].wait();
    auto end = clock::now();
    double elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() / static_cast<double>(runs);
    state.counters["flops"] = ops / elapsed_seconds;
    state.counters["throughput"] = bytes_transfered / elapsed_seconds;
    state.SetIterationTime(elapsed_seconds);
  }
}

// Helper functions for GBench
template <typename... Args>
void real_time_complex_float(Args&&... args) {
  bench_dft_real_time<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>(
      std::forward<Args>(args)...);
}

template <typename... Args>
void real_time_float(Args&&... args) {
  bench_dft_real_time<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>(std::forward<Args>(args)...);
}

template <typename... Args>
void device_time_complex_float(Args&&... args) {
  bench_dft_device_time<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>(
      std::forward<Args>(args)...);
}

template <typename... Args>
void device_time_float(Args&&... args) {
  bench_dft_device_time<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>(
      std::forward<Args>(args)...);
}

constexpr std::size_t runs_to_average = 10;

template <typename... Args>
void average_host_time_complex_float(Args&&... args) {
  bench_dft_average_host_time<runs_to_average, oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>(
      std::forward<Args>(args)...);
}

template <typename... Args>
void average_host_time_float(Args&&... args) {
  bench_dft_average_host_time<runs_to_average, oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>(
      std::forward<Args>(args)...);
}

#define BENCH_COMPLEX_FLOAT(...)                                                    \
  BENCHMARK_CAPTURE(real_time_complex_float, __VA_ARGS__)->UseManualTime();         \
  BENCHMARK_CAPTURE(average_host_time_complex_float, __VA_ARGS__)->UseManualTime(); \
  BENCHMARK_CAPTURE(device_time_complex_float, __VA_ARGS__)->UseManualTime();

#define BENCH_SINGLE_FLOAT(...)                                             \
  BENCHMARK_CAPTURE(real_time_float, __VA_ARGS__)->UseManualTime();         \
  BENCHMARK_CAPTURE(average_host_time_float, __VA_ARGS__)->UseManualTime(); \
  BENCHMARK_CAPTURE(device_time_float, __VA_ARGS__)->UseManualTime();

INSTANTIATE_REFERENCE_BENCHMARK_SET(BENCH_COMPLEX_FLOAT, BENCH_SINGLE_FLOAT);

BENCHMARK_MAIN();
