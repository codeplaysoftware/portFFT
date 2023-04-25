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
 *  Benchmark of rocFFT for comparison with SYCL-FFT.
 *
 **************************************************************************/

#include <functional>
#include <iostream>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include <rocfft.h>

#include <benchmark/benchmark.h>

#include "bench_utils.hpp"
#include "ops_estimate.hpp"
#include "reference_dft.hpp"
#include "reference_dft_set.hpp"
#include "rocfft_utils.hpp"

template <typename forward_type>
using is_real = std::bool_constant<std::is_same_v<forward_type, float> || std::is_same_v<forward_type, double>>;

template <typename forward_type>
using is_forward_type =
    std::bool_constant<is_real<forward_type>::value || std::is_same_v<forward_type, std::complex<float>> ||
                       std::is_same_v<forward_type, std::complex<double>>>;

template <typename forward_type>
using backward_type = std::conditional<is_real<forward_type>::value, std::complex<forward_type>, forward_type>;

template <typename forward_type>
using is_double =
    std::bool_constant<std::is_same_v<forward_type, double> || std::is_same_v<forward_type, std::complex<double>>>;

template <typename forward_type>
std::size_t get_backward_row_size(const std::vector<std::size_t>& lengths) {
  if constexpr (is_real<forward_type>::value) {
    return lengths.back() / 2 + 1;
  } else {
    return lengths.back();
  }
}

std::size_t get_num_rows(const std::vector<std::size_t>& lengths) {
  return std::reduce(lengths.begin(), lengths.end() - 1, 1, std::multiplies<>());
}

// the number of elements in the backwards domain of a single transform
template <typename forward_type>
std::size_t get_backward_elements(const std::vector<std::size_t>& lengths) {
  return get_num_rows(lengths) * get_backward_row_size<forward_type>(lengths);
}

template <typename forward_type>
void verify_dft(forward_type* forward_copy, void* dev_bwd, const std::vector<size_t>& lengths,
                std::size_t number_of_transforms) {
  using bwd_type = typename backward_type<forward_type>::type;
  std::size_t fwd_per_transform = std::accumulate(lengths.begin(), lengths.end(), 1, std::multiplies<std::size_t>());
  std::size_t fwd_row_size = lengths.back();
  std::size_t bwd_row_size = get_backward_row_size<forward_type>(lengths);
  std::size_t rows = get_num_rows(lengths);
  std::size_t bwd_per_transform = get_backward_elements<forward_type>(lengths);
  std::size_t bwd_elements = number_of_transforms * bwd_per_transform;

  auto host_bwd = std::make_unique<bwd_type[]>(bwd_elements);
  HIP_CHECK(hipMemcpy(host_bwd.get(), dev_bwd, bwd_elements * sizeof(bwd_type), hipMemcpyDeviceToHost));

  HIP_CHECK(hipStreamSynchronize(nullptr));

  std::vector<int> int_lengths(lengths.size());
  std::copy(lengths.begin(), lengths.end(), int_lengths.begin());

  auto reference_buffer = std::make_unique<bwd_type[]>(fwd_per_transform);
  constexpr double comparison_tolerance = 1e-2;
  for (std::size_t i = 0; i < number_of_transforms; ++i) {
    // generate reference for a single transform
    reference_dft<sycl_fft::direction::FORWARD>(forward_copy + i * fwd_per_transform, reference_buffer.get(),
                                                int_lengths);

    // compare
    if (!compare_result(reference_buffer.get(), host_bwd.get() + i * bwd_per_transform, lengths, comparison_tolerance,
                        is_real<forward_type>::value)) {
      throw std::runtime_error("Verification Failed");
    }
  }
}

template <typename forward_type>
struct rocfft_state {
  static_assert(is_forward_type<forward_type>::value, "unexpected forward type");
  benchmark::State& state;
  rocfft_plan plan = nullptr;
  rocfft_execution_info info = nullptr;
  void* fwd = nullptr;
  void* bwd = nullptr;
  void* work_buf = nullptr;
  std::size_t work_buf_size = 0;

  rocfft_state(benchmark::State& state, const std::vector<std::size_t>& lengths, std::size_t number_of_transforms)
      : state(state) {
    // setup rocfft
    ROCFFT_CHECK(rocfft_setup());

    // plan information
    const auto placement = rocfft_placement_notinplace;
    const auto transform_type =
        is_real<forward_type>::value ? rocfft_transform_type_real_forward : rocfft_transform_type_complex_forward;
    const auto precision = is_double<forward_type>::value ? rocfft_precision_double : rocfft_precision_single;

    // initialise plan
    ROCFFT_CHECK(rocfft_plan_create(&plan, placement, transform_type, precision, lengths.size(), lengths.data(),
                                    number_of_transforms, nullptr));

    // plan work buffer
    ROCFFT_CHECK(rocfft_plan_get_work_buffer_size(plan, &work_buf_size));
    if (work_buf_size != 0) {
      HIP_CHECK(hipMalloc(&work_buf, work_buf_size));
      ROCFFT_CHECK(rocfft_execution_info_create(&info));
      ROCFFT_CHECK(rocfft_execution_info_set_work_buffer(info, work_buf, work_buf_size));
    }

    // data buffers
    const std::size_t forward_elements =
        std::reduce(lengths.begin(), lengths.end(), number_of_transforms, std::multiplies<>());

    const std::size_t backward_elements = number_of_transforms * get_backward_elements<forward_type>(lengths);

    HIP_CHECK(hipMalloc(&fwd, forward_elements * sizeof(forward_type)));
    HIP_CHECK(hipMalloc(&bwd, backward_elements * sizeof(typename backward_type<forward_type>::type)));

#ifdef SYCLFFT_VERIFY_BENCHMARK
    using scalar_type = typename backward_type<forward_type>::type::value_type;
    roc_populate_with_random(static_cast<scalar_type*>(fwd),
                             (forward_elements * sizeof(forward_type)) / sizeof(scalar_type));
#endif
  }

  ~rocfft_state() {
    HIP_CHECK_NO_THROW(hipFree(fwd));
    HIP_CHECK_NO_THROW(hipFree(bwd));
    if (work_buf_size != 0) {
      HIP_CHECK_NO_THROW(hipFree(work_buf));
      ROCFFT_CHECK_NO_THROW(rocfft_execution_info_destroy(info));
    }

    ROCFFT_CHECK_NO_THROW(rocfft_plan_destroy(plan));

    ROCFFT_CHECK_NO_THROW(rocfft_cleanup());
  }
};

template <typename forward_type>
void rocfft_oop_real_time(benchmark::State& state, std::vector<int> lengths, int batch) {
  std::vector<std::size_t> roc_lengths(lengths.size());
  std::copy(lengths.begin(), lengths.end(), roc_lengths.begin());
  rocfft_state<forward_type> roc_state(state, roc_lengths, batch);

  rocfft_plan plan = roc_state.plan;
  rocfft_execution_info info = roc_state.info;
  void* in = roc_state.fwd;
  void* out = roc_state.bwd;
  const auto fft_size = std::accumulate(roc_lengths.begin(), roc_lengths.end(), 1, std::multiplies<std::size_t>());
  const auto ops_est = cooley_tukey_ops_estimate(fft_size, batch);
  const auto mem_transactions =
      global_mem_transactions<forward_type, typename backward_type<forward_type>::type>(fft_size, batch);

#ifdef SYCLFFT_VERIFY_BENCHMARK
  // rocfft modifies the input values, so for validation we need to save them before the run
  const auto N = fft_size * batch;
  auto fwd_copy = std::make_unique<forward_type[]>(N);
  HIP_CHECK(hipMemcpy(fwd_copy.get(), in, N * sizeof(forward_type), hipMemcpyDeviceToHost));
#endif

  ROCFFT_CHECK(rocfft_execute(plan, &in, &out, info));
  HIP_CHECK(hipStreamSynchronize(nullptr));

#ifdef SYCLFFT_VERIFY_BENCHMARK
  verify_dft<forward_type>(fwd_copy.get(), out, roc_lengths, batch);
#endif

  // benchmark
  for (auto _ : state) {
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    std::ignore = rocfft_execute(plan, &in, &out, info);
    std::ignore = hipStreamSynchronize(nullptr);
    auto end = clock::now();

    double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    state.SetIterationTime(seconds);
    state.counters["flops"] = ops_est / seconds;
    state.counters["bandwidth"] = mem_transactions / seconds;
  }
}

template <typename forward_type>
static void rocfft_oop_device_time(benchmark::State& state, std::vector<int> lengths, int batch) {
  std::vector<std::size_t> roc_lengths(lengths.size());
  std::copy(lengths.begin(), lengths.end(), roc_lengths.begin());
  rocfft_state<forward_type> roc_state(state, roc_lengths, batch);

  rocfft_plan plan = roc_state.plan;
  rocfft_execution_info info = roc_state.info;
  void* in = roc_state.fwd;
  void* out = roc_state.bwd;
  const auto fft_size = std::accumulate(roc_lengths.begin(), roc_lengths.end(), 1, std::multiplies<std::size_t>());
  const auto ops_est = cooley_tukey_ops_estimate(fft_size, batch);
  const auto mem_transactions =
      global_mem_transactions<forward_type, typename backward_type<forward_type>::type>(fft_size, batch);

#ifdef SYCLFFT_VERIFY_BENCHMARK
  // rocfft modifies the input values, so for validation we need to save them before the run
  const auto N = fft_size * batch;
  auto fwd_copy = std::make_unique<forward_type[]>(N);
  HIP_CHECK(hipMemcpy(fwd_copy.get(), in, N * sizeof(forward_type), hipMemcpyDeviceToHost));
#endif

  ROCFFT_CHECK(rocfft_execute(plan, &in, &out, info));
  HIP_CHECK(hipStreamSynchronize(nullptr));

#ifdef SYCLFFT_VERIFY_BENCHMARK
  verify_dft<forward_type>(fwd_copy.get(), out, roc_lengths, batch);
#endif

  hipEvent_t before;
  hipEvent_t after;
  HIP_CHECK(hipEventCreate(&before));
  HIP_CHECK(hipEventCreate(&after));

  // benchmark
  for (auto _ : state) {
    auto before_res = hipEventRecord(before);
    auto exec_res = rocfft_execute(plan, &in, &out, info);
    auto after_res = hipEventRecord(after);
    auto sync_res = hipEventSynchronize(after);
    if (before_res != hipSuccess || exec_res != rocfft_status_success || after_res != hipSuccess ||
        sync_res != hipSuccess) {
      throw std::runtime_error("benchmark run failed");
      return;
    }
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, before, after));
    double seconds = ms / 1000.0;
    state.SetIterationTime(seconds);
    state.counters["flops"] = ops_est / seconds;
    state.counters["bandwidth"] = mem_transactions / seconds;
  }

  HIP_CHECK(hipEventDestroy(before));
  HIP_CHECK(hipEventDestroy(after));
}

// Helper functions for GBench
template <typename... Args>
void rocfft_oop_real_time_complex_float(Args&&... args) {
  rocfft_oop_real_time<std::complex<float>>(std::forward<Args>(args)...);
}

template <typename... Args>
void rocfft_oop_real_time_float(Args&&... args) {
  rocfft_oop_real_time<float>(std::forward<Args>(args)...);
}

template <typename... Args>
void rocfft_oop_device_time_complex_float(Args&&... args) {
  rocfft_oop_device_time<std::complex<float>>(std::forward<Args>(args)...);
}

template <typename... Args>
void rocfft_oop_device_time_float(Args&&... args) {
  rocfft_oop_device_time<float>(std::forward<Args>(args)...);
}

#define BENCH_COMPLEX_FLOAT(...)                                                       \
  BENCHMARK_CAPTURE(rocfft_oop_real_time_complex_float, __VA_ARGS__)->UseManualTime(); \
  BENCHMARK_CAPTURE(rocfft_oop_device_time_complex_float, __VA_ARGS__)->UseManualTime();

#define BENCH_SINGLE_FLOAT(...)                                                \
  BENCHMARK_CAPTURE(rocfft_oop_real_time_float, __VA_ARGS__)->UseManualTime(); \
  BENCHMARK_CAPTURE(rocfft_oop_device_time_float, __VA_ARGS__)->UseManualTime();

INSTANTIATE_REFERENCE_BENCHMARK_SET(BENCH_COMPLEX_FLOAT, BENCH_SINGLE_FLOAT);

BENCHMARK_MAIN();
