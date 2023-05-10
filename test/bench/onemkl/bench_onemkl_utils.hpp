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
#include "device_number_generator.hpp"
#include "number_generators.hpp"
#include "ops_estimate.hpp"
#include "reference_dft_set.hpp"

/// Get the floating-point type from the MKL precision enum.
template <oneapi::mkl::dft::precision prec>
using get_float_t = std::conditional_t<prec == oneapi::mkl::dft::precision::SINGLE, float, double>;

template <oneapi::mkl::dft::domain domain, typename float_t>
using get_forward_t = std::conditional_t<domain == oneapi::mkl::dft::domain::REAL, float_t, std::complex<float_t>>;

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
  for (std::size_t i{0}; i < in_vec.size(); ++i) {
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
  using forward_t = get_forward_t<domain, float_t>;
  using backward_t = std::complex<float_t>;

  // Queue & allocations for test
  descriptor_t desc;
  sycl::queue sycl_queue;
  forward_t* in_dev = nullptr;
  backward_t* out_dev = nullptr;

  // Descriptor data to avoid having to use get_value.
  std::vector<std::int64_t> lengths;
  std::int64_t number_of_transforms;
  std::int64_t fwd_per_transform;
  std::int64_t bwd_per_transform;

  // Constructor. Allocates required memory.
  onemkl_state(sycl::queue sycl_queue, std::vector<std::int64_t> lengths, std::int64_t number_of_transforms)
      : desc(lengths),
        sycl_queue(sycl_queue),
        lengths{lengths},
        number_of_transforms{number_of_transforms},
        fwd_per_transform(get_fwd_per_transform(lengths)),
        bwd_per_transform(get_bwd_per_transform<get_forward_t<domain, get_float_t<prec>>>(lengths)) {
    using config_param_t = oneapi::mkl::dft::config_param;
    // For now, out-of-place only.
    set_out_of_place();
    desc.set_value(config_param_t::NUMBER_OF_TRANSFORMS, number_of_transforms);
    desc.set_value(config_param_t::FWD_DISTANCE, fwd_per_transform);
    desc.set_value(config_param_t::BWD_DISTANCE, bwd_per_transform);

    // strides
    std::array<std::int64_t, 4> strides{};

    // work backwards to generate the strides, since for n>0 stride[n+1] = strides[n]/lengths[n]
    std::size_t idx = lengths.size();
    strides[idx] = 1;
    while (idx != 1) {
      strides[idx - 1] = lengths[idx - 1] * strides[idx];
      idx -= 1;
    }

    desc.set_value(config_param_t::INPUT_STRIDES, &strides);

    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
      // strides must be adjusted to account for the conjugate symmetry
      for (std::size_t i = 1; i < lengths.size(); ++i) {
        strides[i] = (strides[i] / lengths.back()) * (lengths.back() / 2 + 1);
      }
    }

    desc.set_value(config_param_t::OUTPUT_STRIDES, &strides);

    // Allocate memory.
    in_dev = sycl::malloc_device<forward_t>(fwd_per_transform * number_of_transforms, sycl_queue);
    out_dev = sycl::malloc_device<backward_t>(bwd_per_transform * number_of_transforms, sycl_queue);
    sycl_queue.wait_and_throw();

#ifdef SYCLFFT_VERIFY_BENCHMARK
    memFill(in_dev, sycl_queue, fwd_per_transform * number_of_transforms);
    sycl_queue.wait_and_throw();
#endif  // SYCLFFT_VERIFY_BENCHMARK
  }

  ~onemkl_state() {
    sycl::free(in_dev, sycl_queue);
    sycl::free(out_dev, sycl_queue);
  }

  // Backend specific methods
  inline sycl::event compute();

 private:
  // Backend specific methods
  void set_out_of_place();
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
  sycl::queue q{};
  auto lengthsI64 = cast_vector_elements<std::int64_t>(lengths);
  onemkl_state<prec, domain> fft_state{q, lengthsI64, number_of_transforms};

  using forward_t = typename decltype(fft_state)::forward_t;
  using backward_t = typename decltype(fft_state)::backward_t;

  double ops = cooley_tukey_ops_estimate(fft_state.fwd_per_transform, fft_state.number_of_transforms);
  std::size_t bytes_transfered = global_mem_transactions<forward_t, backward_t>(
      fft_state.number_of_transforms, fft_state.fwd_per_transform, fft_state.bwd_per_transform);

#ifdef SYCLFFT_VERIFY_BENCHMARK
  std::vector<forward_t> host_input(fft_state.fwd_per_transform * fft_state.number_of_transforms);
  q.copy<forward_t>(fft_state.in_dev, host_input.data(), host_input.size()).wait_and_throw();
#endif  // SYCLFFT_VERIFY_BENCHMARK

  try {
    fft_state.desc.commit(q);
    q.wait_and_throw();
    // warmup
    fft_state.compute().wait_and_throw();
  } catch (...) {
    // Can't run this benchmark!
    state.SkipWithError("Exception thrown: commit or warm-up failed.");
    return;
  }

#ifdef SYCLFFT_VERIFY_BENCHMARK
  std::vector<backward_t> host_output(fft_state.bwd_per_transform * fft_state.number_of_transforms);
  q.copy<backward_t>(fft_state.out_dev, host_output.data(), host_output.size()).wait_and_throw();
  verify_dft<forward_t, backward_t>(host_input.data(), host_output.data(), lengths, number_of_transforms, 1.0);
#endif  // SYCLFFT_VERIFY_BENCHMARK

  for (auto _ : state) {
    // we need to manually measure time, so as to have it available here for the
    // calculation of flops
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    fft_state.compute().wait();
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
  // Get key information out of the descriptor.
  sycl::queue q({sycl::property::queue::enable_profiling()});
  const auto lengthsI64 = cast_vector_elements<std::int64_t>(lengths);
  onemkl_state<prec, domain> fft_state{q, lengthsI64, number_of_transforms};

  using forward_t = typename decltype(fft_state)::forward_t;
  using backward_t = typename decltype(fft_state)::backward_t;

  const double ops = cooley_tukey_ops_estimate(fft_state.fwd_per_transform, fft_state.number_of_transforms);
  const std::size_t bytes_transfered = global_mem_transactions<forward_t, backward_t>(
      fft_state.number_of_transforms, fft_state.fwd_per_transform, fft_state.bwd_per_transform);

#ifdef SYCLFFT_VERIFY_BENCHMARK
  std::vector<forward_t> host_input(fft_state.fwd_per_transform * fft_state.number_of_transforms);
  q.copy(fft_state.in_dev, host_input.data(), host_input.size()).wait_and_throw();
#endif  // SYCLFFT_VERIFY_BENCHMARK

  try {
    fft_state.desc.commit(q);
    q.wait_and_throw();
    // warmup
    fft_state.compute().wait_and_throw();
  } catch (...) {
    state.SkipWithError("Exception thrown: commit or warm-up failed.");
    return;
  }

#ifdef SYCLFFT_VERIFY_BENCHMARK
  std::vector<backward_t> host_output(fft_state.bwd_per_transform * fft_state.number_of_transforms);
  q.copy(fft_state.out_dev, host_output.data(), host_output.size()).wait_and_throw();
  verify_dft(host_input.data(), host_output.data(), lengths, number_of_transforms, 1.0);
#endif  // SYCLFFT_VERIFY_BENCHMARK

  for (auto _ : state) {
    int64_t start{0}, end{0};
    try {
      sycl::event e = fft_state.compute();
      e.wait();
      start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
      end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    } catch (sycl::exception& e) {
      // e may not have profiling info, so this benchmark is useless
      auto errorMessage = std::string("Exception thrown ") + e.what();
      state.SkipWithError(errorMessage.c_str());
      return;
    }
    double elapsed_seconds = (end - start) / 1e9;
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

#define BENCH_COMPLEX_FLOAT(...)                                            \
  BENCHMARK_CAPTURE(real_time_complex_float, __VA_ARGS__)->UseManualTime(); \
  BENCHMARK_CAPTURE(device_time_complex_float, __VA_ARGS__)->UseManualTime();

#define BENCH_SINGLE_FLOAT(...)                                     \
  BENCHMARK_CAPTURE(real_time_float, __VA_ARGS__)->UseManualTime(); \
  BENCHMARK_CAPTURE(device_time_float, __VA_ARGS__)->UseManualTime();

INSTANTIATE_REFERENCE_BENCHMARK_SET(BENCH_COMPLEX_FLOAT, BENCH_SINGLE_FLOAT);

BENCHMARK_MAIN();
