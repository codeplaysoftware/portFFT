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
 *  Benchmark of OneMKL for comparison with SYCL-FFT.
 *
 *  Building with closed-source MKL:
 *  Set path to MKL_ROOT and set
 *  SYCLFFT_ENABLE_INTEL_CLOSED_ONEMKL_BENCHMARKS=ON.
 *
 **************************************************************************/

#include <complex>
#include <type_traits>
#include <vector>

// Intel's closed-source OneMKL library header.
#include <oneapi/mkl/dfti.hpp>
#include <oneapi/mkl/exceptions.hpp>

#include <benchmark/benchmark.h>

#include "number_generators.hpp"
#include "ops_estimate.hpp"

/// Get the floating-point type from the MKL precision enum.
template <oneapi::mkl::dft::precision prec>
using get_float_t =
    std::conditional_t<prec == oneapi::mkl::dft::precision::SINGLE, float,
                       double>;

/// Copy an input vector to an output vector, with element-wise casts.
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
  onemkl_state(sycl::queue sycl_queue, std::vector<std::int64_t> lengths,
               std::int64_t number_of_transforms)
      : desc(lengths),
        sycl_queue(sycl_queue),
        lengths{lengths},
        number_of_transforms{number_of_transforms} {
    using config_param_t = oneapi::mkl::dft::config_param;
    // For now, out-of-place only.
    desc.set_value(config_param_t::PLACEMENT, DFTI_NOT_INPLACE);
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

  inline sycl::event compute() {
    return compute_forward(desc, in_dev, out_dev);
  }

  /// The count of bytes for each FFT. Product of lengths.
  inline std::size_t get_total_length() {
    return std::accumulate(lengths.cbegin(), lengths.cend(), 1,
                           std::multiplies<>());
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
void bench_dft_real_time(benchmark::State& state, std::vector<int> lengths,
                         int number_of_transforms) {
  using float_type = get_float_t<prec>;
  using complex_type = std::complex<float_type>;
  sycl::queue q;
  auto lengthsI64 = cast_vector_elements<std::int64_t>(lengths);
  onemkl_state<prec, domain> fft_state{q, lengthsI64, number_of_transforms};
  std::size_t N = fft_state.get_total_length();
  double ops = cooley_tukey_ops_estimate(N, fft_state.number_of_transforms);
  std::vector<complex_type> a(fft_state.num_elements);
  populate_with_random(a);

  q.copy(a.data(), fft_state.in_dev, fft_state.num_elements);

  try {
    fft_state.desc.commit(q);
    q.wait_and_throw();
    // warmup
    fft_state.compute().wait_and_throw();
  } catch (sycl::_V1::runtime_error&) {
    // Can't run this benchmark!
    return;
  }

  for (auto _ : state) {
    // we need to manually measure time, so as to have it available here for the
    // calculation of flops
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    fft_state.compute().wait();
    auto end = clock::now();
    double elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
    state.counters["flops"] = ops / elapsed_seconds;
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
void bench_dft_device_time(benchmark::State& state, std::vector<int> lengths,
                           int number_of_transforms) {
  using float_type = get_float_t<prec>;
  using complex_type = std::complex<float_type>;
  // Get key information out of the descriptor.
  sycl::queue q({sycl::property::queue::enable_profiling()});
  auto lengthsI64 = cast_vector_elements<std::int64_t>(lengths);
  onemkl_state<prec, domain> fft_state{q, lengthsI64, number_of_transforms};
  std::size_t N = fft_state.get_total_length();
  double ops = cooley_tukey_ops_estimate(N, fft_state.number_of_transforms);
  std::vector<complex_type> a(fft_state.num_elements);
  populate_with_random(a);

  q.copy(a.data(), fft_state.in_dev, fft_state.num_elements);

  try {
    fft_state.desc.commit(q);
    q.wait_and_throw();
    // warmup
    fft_state.compute().wait_and_throw();
  } catch (sycl::_V1::runtime_error&) {
    // Can't run this benchmark!
    return;
  }

  for (auto _ : state) {
    int64_t start{0}, end{0};
    try {
      sycl::event e = fft_state.compute();
      e.wait();
      start =
          e.get_profiling_info<sycl::info::event_profiling::command_start>();
      end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    } catch (sycl::_V1::runtime_error&) {
      // e may not have profiling info, so this benchmark is useless
      start = end;
    }
    double elapsed_seconds = (end - start) / 1e9;
    state.counters["flops"] = ops / elapsed_seconds;
    state.SetIterationTime(elapsed_seconds);
  }
}

// Helper functions for GBench
template <typename... Args>
void real_time_complex_float(Args&&... args) {
  bench_dft_real_time<oneapi::mkl::dft::precision::SINGLE,
                      oneapi::mkl::dft::domain::COMPLEX>(
      std::forward<Args>(args)...);
}

template <typename... Args>
void real_time_float(Args&&... args) {
  bench_dft_real_time<oneapi::mkl::dft::precision::SINGLE,
                      oneapi::mkl::dft::domain::REAL>(
      std::forward<Args>(args)...);
}

template <typename... Args>
void device_time_complex_float(Args&&... args) {
  bench_dft_device_time<oneapi::mkl::dft::precision::SINGLE,
                        oneapi::mkl::dft::domain::COMPLEX>(
      std::forward<Args>(args)...);
}

template <typename... Args>
void device_time_float(Args&&... args) {
  bench_dft_device_time<oneapi::mkl::dft::precision::SINGLE,
                        oneapi::mkl::dft::domain::REAL>(
      std::forward<Args>(args)...);
}

#define BENCH_COMPLEX_FLOAT(...)                           \
  BENCHMARK_CAPTURE(real_time_complex_float, __VA_ARGS__); \
  BENCHMARK_CAPTURE(device_time_complex_float, __VA_ARGS__)

#define BENCH_SINGLE_FLOAT(...)                    \
  BENCHMARK_CAPTURE(real_time_float, __VA_ARGS__); \
  BENCHMARK_CAPTURE(device_time_float, __VA_ARGS__)

#include "reference_dft_set.cxx"

BENCHMARK_MAIN();
