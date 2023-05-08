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
#include "sycl_utils.hpp"

template <typename Backward, oneapi::mkl::dft::precision Prec, oneapi::mkl::dft::domain Domain>
struct forward_type_info_impl {
  using backward_type = Backward;
  static constexpr oneapi::mkl::dft::precision prec = Prec;
  static constexpr oneapi::mkl::dft::domain domain = Domain;
};

template <typename T>
struct forward_type_info;
template <>
struct forward_type_info<float>
    : forward_type_info_impl<std::complex<float>, oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL> {
};
template <>
struct forward_type_info<std::complex<float>>
    : forward_type_info_impl<std::complex<float>, oneapi::mkl::dft::precision::SINGLE,
                             oneapi::mkl::dft::domain::COMPLEX> {};
template <>
struct forward_type_info<double> : forward_type_info_impl<std::complex<double>, oneapi::mkl::dft::precision::DOUBLE,
                                                          oneapi::mkl::dft::domain::REAL> {};
template <>
struct forward_type_info<std::complex<double>>
    : forward_type_info_impl<std::complex<double>, oneapi::mkl::dft::precision::DOUBLE,
                             oneapi::mkl::dft::domain::COMPLEX> {};

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
template <typename forward_type>
struct onemkl_state {
  using type_info = forward_type_info<forward_type>;
  using backward_type = typename type_info::backward_type;
  using descriptor_type = oneapi::mkl::dft::descriptor<type_info::prec, type_info::domain>;

  // Queue & allocations for test
  descriptor_type desc;
  sycl::queue sycl_queue;
  forward_type* in_dev = nullptr;
  backward_type* out_dev = nullptr;

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
        bwd_per_transform(get_bwd_per_transform<forward_type>(lengths)) {
    using config_param_t = oneapi::mkl::dft::config_param;
    // For now, out-of-place only.
    set_out_of_place();
    desc.set_value(config_param_t::NUMBER_OF_TRANSFORMS, number_of_transforms);
    desc.set_value(config_param_t::FWD_DISTANCE, fwd_per_transform);
    desc.set_value(config_param_t::BWD_DISTANCE, bwd_per_transform);

    // strides
    std::array<std::int64_t, 4> strides{0, 0, 0, 0};

    // work backwards to generate the strides, since for n>0 stride[n+1] = strides[n]/lengths[n]
    std::size_t idx = lengths.size();
    strides[idx] = 1;
    while (idx != 1) {
      strides[idx - 1] = lengths[idx - 1] * strides[idx];
      idx -= 1;
    }

    desc.set_value(config_param_t::INPUT_STRIDES, &strides);

    if constexpr (type_info::domain == oneapi::mkl::dft::domain::REAL) {
      // strides must be adjusted to account for the conjugate symmetry
      for (std::size_t i = 1; i < lengths.size(); ++i) {
        strides[i] = (strides[i] / lengths.back()) * (lengths.back() / 2 + 1);
      }
    }

    desc.set_value(config_param_t::OUTPUT_STRIDES, &strides);

    // Allocate memory.
    in_dev = sycl::malloc_device<forward_type>(fwd_per_transform * number_of_transforms, sycl_queue);
    out_dev = sycl::malloc_device<backward_type>(bwd_per_transform * number_of_transforms, sycl_queue);
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
  inline sycl::event compute(const std::vector<sycl::event>& deps = {});

 private:
  // Backend specific methods
  void set_out_of_place();
};

/**
 * @brief Main function to run oneMKL benchmarks and measure the time spent on the host.
 * One GBench iteration consists of multiple compute submitted asynchronously to reduce the overhead of the SYCL
 * runtime. The function throws exception if an error occurs.
 *
 * @tparam forward_type Can be std::complex or real, float or double.
 * @param state GBench state.
 * @param q SYCL queue.
 * @param lengths The lengths defining and N-dimensional DFT.
 * @param number_of_transforms The DFT batch size.
 * @param runs The number of runs to average for one GBench iteration.
 */
template <typename forward_type>
void onemkl_average_host_time_impl(benchmark::State& state, sycl::queue q, std::vector<int> lengths,
                                   int number_of_transforms, std::size_t runs) {
  auto lengthsI64 = cast_vector_elements<std::int64_t>(lengths);
  onemkl_state<forward_type> fft_state{q, lengthsI64, number_of_transforms};
  using info = typename decltype(fft_state)::type_info;
  using backward_type = typename info::backward_type;

  double ops = cooley_tukey_ops_estimate(fft_state.fwd_per_transform, fft_state.number_of_transforms);
  std::size_t bytes_transferred = global_mem_transactions<forward_type, backward_type>(
      fft_state.number_of_transforms, fft_state.fwd_per_transform, fft_state.bwd_per_transform);

#ifdef SYCLFFT_VERIFY_BENCHMARK
  std::vector<forward_type> host_input(fft_state.fwd_per_transform * fft_state.number_of_transforms);
  q.copy<forward_type>(fft_state.in_dev, host_input.data(), host_input.size()).wait_and_throw();
#endif  // SYCLFFT_VERIFY_BENCHMARK

  std::vector<sycl::event> dependencies;
  dependencies.reserve(1);

  fft_state.desc.commit(q);
  q.wait_and_throw();
  // warmup
  fft_state.compute().wait_and_throw();

#ifdef SYCLFFT_VERIFY_BENCHMARK
  std::vector<backward_type> host_output(fft_state.bwd_per_transform * fft_state.number_of_transforms);
  q.copy<backward_type>(fft_state.out_dev, host_output.data(), host_output.size()).wait_and_throw();
  verify_dft<forward_type, backward_type>(host_input.data(), host_output.data(), lengths, number_of_transforms, 1.0);
#endif  // SYCLFFT_VERIFY_BENCHMARK

  for (auto _ : state) {
    // we need to manually measure time, so as to have it available here for the
    // calculation of flops

    dependencies.clear();
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    dependencies.emplace_back(fft_state.compute());
    for (std::size_t r = 1; r != runs; r += 1) {
      dependencies[0] = fft_state.compute(dependencies);
    }
    dependencies[0].wait();
    auto end = clock::now();
    double elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() / static_cast<double>(runs);
    state.counters["flops"] = ops / elapsed_seconds;
    state.counters["throughput"] = bytes_transferred / elapsed_seconds;
    state.SetIterationTime(elapsed_seconds);
  }
}

/**
 * @brief Separate impl function to handle errors
 * @see onemkl_average_host_time_impl
 */
template <typename forward_type>
void onemkl_average_host_time(benchmark::State& state, sycl::queue q, std::vector<int> lengths,
                              int number_of_transforms) {
  try {
    onemkl_average_host_time_impl<forward_type>(state, q, lengths, number_of_transforms, runs_to_average);
  } catch (std::exception& e) {
    handle_exception(state, e);
  }
}

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  sycl::queue q;
  print_device(q);
  register_complex_float_benchmark_set("average_host_time", onemkl_average_host_time<std::complex<float>>, q);
  register_real_float_benchmark_set("average_host_time", onemkl_average_host_time<float>, q);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
