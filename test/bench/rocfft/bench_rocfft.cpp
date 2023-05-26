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

#include <chrono>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include <rocfft.h>

#include <benchmark/benchmark.h>

#include "bench_utils.hpp"
#include "ops_estimate.hpp"
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

static void print_device() {
  int device_id;
  HIP_CHECK(hipGetDevice(&device_id));

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
  std::string compute_capabilities = std::to_string(prop.major) + "." + std::to_string(prop.minor);

  int runtime_version;
  HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
  int runtime_major = runtime_version / 1000;
  int runtime_minor = (runtime_version % 1000) / 10;
  std::string runtime_version_str = std::to_string(runtime_major) + "." + std::to_string(runtime_minor);

  // Minimum buffer length required by the rocFFT documentation.
  char rocfft_version[30];
  ROCFFT_CHECK(rocfft_get_version_string(rocfft_version, sizeof(rocfft_version)));

  benchmark::AddCustomContext("Device id", std::to_string(device_id));
  benchmark::AddCustomContext("Name", prop.name);
  benchmark::AddCustomContext("Compute capabilities", compute_capabilities);
  benchmark::AddCustomContext("Runtime version", runtime_version_str);
  benchmark::AddCustomContext("rocFFT version", rocfft_version);
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
  std::size_t fwd_per_transform;
  std::size_t bwd_per_transform;

  rocfft_state(benchmark::State& state, const std::vector<std::size_t>& lengths, std::size_t number_of_transforms)
      : state(state),
        fwd_per_transform(get_fwd_per_transform(lengths)),
        bwd_per_transform(get_bwd_per_transform<forward_type>(lengths)) {
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
    const std::size_t forward_elements = fwd_per_transform * number_of_transforms;
    const std::size_t backward_elements = bwd_per_transform * number_of_transforms;

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

/**
 * @brief Main function to run rocFFT benchmarks and measure the time spent on the host.
 * One GBench iteration consists of multiple compute submitted asynchronously to reduce the overhead of the SYCL
 * runtime. The function throws exception if an error occurs.
 *
 * @tparam forward_type Can be std::complex or real, float or double
 * @param state GBench state
 * @param lengths FFT lengths
 * @param batch FFT batch size
 * @param runs Number of asynchronous compute in one GBench iteration
 */
template <typename forward_type>
void rocfft_oop_average_host_time_impl(benchmark::State& state, std::vector<int> lengths, int batch, std::size_t runs) {
  using backward_t = typename backward_type<forward_type>::type;

  std::vector<std::size_t> roc_lengths(lengths.size());
  std::copy(lengths.begin(), lengths.end(), roc_lengths.begin());
  rocfft_state<forward_type> roc_state(state, roc_lengths, batch);

  rocfft_plan plan = roc_state.plan;
  rocfft_execution_info info = roc_state.info;
  void* in = roc_state.fwd;
  void* out = roc_state.bwd;
  const auto ops_est = cooley_tukey_ops_estimate(roc_state.fwd_per_transform, batch);
  const auto bytes_transferred = global_mem_transactions<forward_type, backward_t>(batch, roc_state.fwd_per_transform,
                                                                                   roc_state.bwd_per_transform);

#ifdef SYCLFFT_VERIFY_BENCHMARK
  // rocfft modifies the input values, so for validation we need to save them before the run
  const auto fwd_elements = roc_state.fwd_per_transform * batch;
  auto fwd_copy = std::make_unique<forward_type[]>(fwd_elements);
  HIP_CHECK(hipMemcpy(fwd_copy.get(), in, fwd_elements * sizeof(forward_type), hipMemcpyDeviceToHost));
#endif

  ROCFFT_CHECK(rocfft_execute(plan, &in, &out, info));
  HIP_CHECK(hipStreamSynchronize(nullptr));

#ifdef SYCLFFT_VERIFY_BENCHMARK
  const auto bwd_elements = roc_state.bwd_per_transform * batch;
  auto bwd_copy = std::make_unique<backward_t[]>(bwd_elements);
  HIP_CHECK(hipMemcpy(bwd_copy.get(), out, bwd_elements * sizeof(backward_t), hipMemcpyDeviceToHost));
  verify_dft<forward_type, backward_t>(fwd_copy.get(), bwd_copy.get(), lengths, batch, 1.0);
#endif

  // benchmark
  for (auto _ : state) {
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    for (std::size_t r = 0; r != runs; r += 1) {
      std::ignore = rocfft_execute(plan, &in, &out, info);
    }
    std::ignore = hipStreamSynchronize(nullptr);
    auto end = clock::now();

    double seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() / static_cast<double>(runs);
    state.SetIterationTime(seconds);
    state.counters["flops"] = ops_est / seconds;
    state.counters["throughput"] = bytes_transferred / seconds;
  }
}

/**
 * @brief Separate impl function to handle
 * @see rocfft_oop_average_host_time_impl
 */
template <typename forward_type>
void rocfft_oop_average_host_time(benchmark::State& state, std::vector<int> lengths, int batch) {
  try {
    rocfft_oop_average_host_time_impl<forward_type>(state, lengths, batch, runs_to_average);
  } catch (std::exception& e) {
    handle_exception(state, e);
  }
}

/**
 * @brief Main function to run rocFFT benchmarks and measure the time spent on the device.
 * The function throws exception if an error occurs.
 *
 * @tparam forward_type Can be std::complex or real, float or double
 * @param state GBench state
 * @param lengths FFT lengths
 * @param batch FFT batch size
 */
template <typename forward_type>
static void rocfft_oop_device_time_impl(benchmark::State& state, std::vector<int> lengths, int batch) {
  using backward_t = typename backward_type<forward_type>::type;

  std::vector<std::size_t> roc_lengths(lengths.size());
  std::copy(lengths.begin(), lengths.end(), roc_lengths.begin());
  rocfft_state<forward_type> roc_state(state, roc_lengths, batch);

  rocfft_plan plan = roc_state.plan;
  rocfft_execution_info info = roc_state.info;
  void* in = roc_state.fwd;
  void* out = roc_state.bwd;
  const auto ops_est = cooley_tukey_ops_estimate(roc_state.fwd_per_transform, batch);
  const auto bytes_transferred = global_mem_transactions<forward_type, backward_t>(batch, roc_state.fwd_per_transform,
                                                                                   roc_state.bwd_per_transform);

#ifdef SYCLFFT_VERIFY_BENCHMARK
  // rocfft modifies the input values, so for validation we need to save them before the run
  const auto fwd_elements = roc_state.fwd_per_transform * batch;
  auto fwd_copy = std::make_unique<forward_type[]>(fwd_elements);
  HIP_CHECK(hipMemcpy(fwd_copy.get(), in, fwd_elements * sizeof(forward_type), hipMemcpyDeviceToHost));
#endif

  ROCFFT_CHECK(rocfft_execute(plan, &in, &out, info));
  HIP_CHECK(hipStreamSynchronize(nullptr));

#ifdef SYCLFFT_VERIFY_BENCHMARK
  const auto bwd_elements = roc_state.bwd_per_transform * batch;
  auto bwd_copy = std::make_unique<backward_t[]>(bwd_elements);
  HIP_CHECK(hipMemcpy(bwd_copy.get(), out, bwd_elements * sizeof(backward_t), hipMemcpyDeviceToHost));
  verify_dft<forward_type, backward_t>(fwd_copy.get(), bwd_copy.get(), lengths, batch, 1.0);
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
    state.counters["throughput"] = bytes_transferred / seconds;
  }

  HIP_CHECK(hipEventDestroy(before));
  HIP_CHECK(hipEventDestroy(after));
}

/**
 * @brief Separate impl function to handle errors
 * @see rocfft_oop_device_time_impl
 */
template <typename forward_type>
static void rocfft_oop_device_time(benchmark::State& state, std::vector<int> lengths, int batch) {
  try {
    rocfft_oop_device_time_impl<forward_type>(state, lengths, batch);
  } catch (std::exception& e) {
    handle_exception(state, e);
  }
}

int main(int argc, char** argv) {
  benchmark::SetDefaultTimeUnit(benchmark::kMillisecond);
  benchmark::Initialize(&argc, argv);
  print_device();
  register_complex_float_benchmark_set("average_host_time", rocfft_oop_average_host_time<std::complex<float>>);
  register_complex_float_benchmark_set("device_time", rocfft_oop_device_time<std::complex<float>>);
  register_real_float_benchmark_set("average_host_time", rocfft_oop_average_host_time<float>);
  register_real_float_benchmark_set("device_time", rocfft_oop_device_time<float>);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
