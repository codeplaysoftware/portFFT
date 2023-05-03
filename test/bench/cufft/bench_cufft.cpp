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
 *  Benchmark of cuFFT for comparison with SYCL-FFT.
 *
 **************************************************************************/

#include <chrono>
#include <complex>
#include <memory>
#include <numeric>
#include <optional>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

#include <benchmark/benchmark.h>

#include "bench_utils.hpp"
#include "cufft_utils.hpp"
#include "enums.hpp"
#include "ops_estimate.hpp"
#include "reference_dft.hpp"
#include "reference_dft_set.hpp"

template <cufftType T>
struct scalar_data_type {
  using type = float;  // default value;
};

template <>
struct scalar_data_type<CUFFT_D2Z> {
  using type = double;
};

template <>
struct scalar_data_type<CUFFT_Z2Z> {
  using type = double;
};

inline int get_forward_fft_size(const std::vector<int>& lengths) {
  return std::accumulate(lengths.begin(), lengths.end(), 1, std::multiplies<int>());
}

template <cufftType plan_type>
inline int get_backward_fft_size(const std::vector<int>& lengths) {
  if constexpr (plan_type == CUFFT_R2C || plan_type == CUFFT_D2Z) {
    return std::accumulate(lengths.begin(), lengths.end() - 1, lengths.back() / 2 + 1, std::multiplies<int>());
  } else {
    return get_forward_fft_size(lengths);
  }
}

template <cufftType plan_type, typename TypeIn, typename TypeOut>
void verify_dft(TypeIn* dev_input, TypeOut* dev_output, const std::vector<int>& lengths, std::size_t batch) {
  std::size_t fft_size = get_forward_fft_size(lengths);
  std::size_t bwd_fft_size = get_backward_fft_size<plan_type>(lengths);

  std::size_t num_elements = batch * fft_size;
  std::vector<TypeIn> host_input(num_elements);
  std::vector<TypeOut> host_output(num_elements);
  cudaMemcpy(host_output.data(), dev_output, num_elements * sizeof(TypeOut), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_input.data(), dev_input, num_elements * sizeof(TypeIn), cudaMemcpyDeviceToHost);

  using scalar_type = typename scalar_data_type<plan_type>::type;
  std::vector<TypeOut> result_vector(fft_size);
  for (std::size_t i = 0; i < batch; i++) {
    if constexpr (std::is_same_v<cufftComplex, TypeIn> || std::is_same_v<cufftDoubleComplex, TypeIn>) {
      reference_dft<sycl_fft::direction::FORWARD>(
          reinterpret_cast<std::complex<scalar_type>*>(host_input.data() + i * fft_size),
          reinterpret_cast<std::complex<scalar_type>*>(result_vector.data()), lengths);
    } else {
      reference_dft<sycl_fft::direction::FORWARD>(host_input.data() + i * fft_size,
                                                  reinterpret_cast<std::complex<scalar_type>*>(result_vector.data()),
                                                  lengths);
    }
    int correct = compare_result(reinterpret_cast<std::complex<scalar_type>*>(result_vector.data()),
                                 reinterpret_cast<std::complex<scalar_type>*>(host_output.data() + i * bwd_fft_size),
                                 lengths, 1e-2, plan_type == CUFFT_R2C);
    if (!correct) {
      throw std::runtime_error("Verification Failed");
    }
  }
}

template <typename Backward, typename DeviceForward, typename DeviceBackward, cufftType plan>
struct forward_type_info_impl {
  using backward_type = Backward;
  using device_forward_type = DeviceForward;
  using device_backward_type = DeviceBackward;
  static constexpr cufftType plan_type = plan;
};

template <typename T>
struct forward_type_info;
template <>
struct forward_type_info<float> : forward_type_info_impl<std::complex<float>, cufftReal, cufftComplex, CUFFT_R2C> {};
template <>
struct forward_type_info<std::complex<float>>
    : forward_type_info_impl<std::complex<float>, cufftComplex, cufftComplex, CUFFT_C2C> {};
template <>
struct forward_type_info<double>
    : forward_type_info_impl<std::complex<double>, cufftDoubleReal, cufftDoubleComplex, CUFFT_D2Z> {};
template <>
struct forward_type_info<std::complex<double>>
    : forward_type_info_impl<std::complex<double>, cufftDoubleComplex, cufftDoubleComplex, CUFFT_Z2Z> {};

template <typename T>
struct cuda_freer {
  benchmark::State& test_state;

  cuda_freer(benchmark::State& s) : test_state(s) {}
  void operator()(T* cu_ptr) {
    if (cudaFree(cu_ptr) != cudaSuccess) {
      test_state.SkipWithError("cudaFree failed");
    }
  }
};

struct cufftHandle_holder {
  benchmark::State& test_state;
  std::optional<cufftHandle> handle;

  cufftHandle_holder(benchmark::State& s, std::optional<cufftHandle> h) : test_state(s), handle(h) {}
  ~cufftHandle_holder() {
    if (handle) {
      if (cufftDestroy(handle.value()) != CUFFT_SUCCESS) {
        test_state.SkipWithError("plan cufftDestroy failed");
      }
    }
  }
};

template <typename forward_type>
struct cufft_state {
  using type_info = forward_type_info<forward_type>;

  benchmark::State& test_state;
  cufftHandle_holder plan;
  std::unique_ptr<typename type_info::device_forward_type, cuda_freer<typename type_info::device_forward_type>> in;
  std::unique_ptr<typename type_info::device_backward_type, cuda_freer<typename type_info::device_backward_type>> out;

  cufft_state(benchmark::State& state, std::vector<int>& lengths, int batch)
      : test_state(state),
        plan(state, {}),
        in(nullptr, cuda_freer<typename type_info::device_forward_type>{state}),
        out(nullptr, cuda_freer<typename type_info::device_backward_type>{state}) {
    if (lengths.empty()) {
      test_state.SkipWithError("invalid configuration");
    }
    int fft_size = get_forward_fft_size(lengths);
    // nullptr inembed and onembed is equivalent to giving the lengths for both
    int *inembed = nullptr, *onembed = nullptr;
    int istride = 1, ostride = 1;
    int idist = fft_size, odist = fft_size;
    cufftHandle plan_tmp;
    auto res = cufftPlanMany(&plan_tmp, lengths.size(), lengths.data(), inembed, istride, idist, onembed, ostride,
                             odist, type_info::plan_type, batch);
    if (res == CUFFT_SUCCESS) {
      plan.handle = plan_tmp;
    } else {
      test_state.SkipWithError("plan creation failed");
    }

    const auto elements = static_cast<std::size_t>(fft_size * batch);
    typename type_info::device_forward_type* in_tmp;
    // TODO overallocing in the REAL-COMPLEX case
    if (cudaMalloc(&in_tmp, sizeof(forward_type) * elements) == cudaSuccess) {
      in.reset(in_tmp);
    } else {
      test_state.SkipWithError("in allocation failed");
    }

    typename type_info::device_backward_type* out_tmp;
    if (cudaMalloc(&out_tmp, sizeof(typename type_info::backward_type) * elements) == cudaSuccess) {
      out.reset(out_tmp);
    } else {
      test_state.SkipWithError("out allocation failed");
    }
#ifdef SYCLFFT_VERIFY_BENCHMARK
    populate_with_random(reinterpret_cast<typename scalar_data_type<type_info::plan_type>::type*>(in.get()), elements);
#endif  // SYCLFFT_VERIFY_BENCHMARK
  }
};

template <typename fwd_type_info>
inline cufftResult cufft_exec(cufftHandle plan, typename fwd_type_info::device_forward_type* in,
                              typename fwd_type_info::device_backward_type* out) noexcept {
  // choose exec function
  if constexpr (fwd_type_info::plan_type == CUFFT_C2C) {
    return cufftExecC2C(plan, in, out, CUFFT_FORWARD);
  } else if constexpr (fwd_type_info::plan_type == CUFFT_Z2Z) {
    return cufftExecZ2Z(plan, in, out, CUFFT_FORWARD);
  } else if constexpr (fwd_type_info::plan_type == CUFFT_R2C) {
    return cufftExecR2C(plan, in, out);
  } else if constexpr (fwd_type_info::plan_type == CUFFT_D2Z) {
    return cufftExecD2Z(plan, in, out);
  }
}

template <typename forward_type>
static void cufft_oop_real_time(benchmark::State& state, std::vector<int> lengths, int batch) noexcept {
  // setup state
  cufft_state<forward_type> cu_state(state, lengths, batch);

  // remove all the extra guff stored in the state
  auto plan = cu_state.plan.handle.value();
  auto in = cu_state.in.get();
  auto out = cu_state.out.get();

  // ops estimate for flops
  const auto fft_size = get_forward_fft_size(lengths);
  const auto ops_est = cooley_tukey_ops_estimate(fft_size, batch);
  using forward_info = typename forward_type_info<forward_type>;
  const int out_size = get_backward_fft_size<forward_info::plan_type>(lengths);
  const auto bytes_transfered =
      global_mem_transactions<typename forward_info::device_forward_type, typename forward_info::device_backward_type>(
          batch, fft_size, out_size);

  // warmup
  if (cufft_exec<typename decltype(cu_state)::type_info>(plan, in, out) != CUFFT_SUCCESS) {
    state.SkipWithError("warmup exec failed");
  }
  if (cudaStreamSynchronize(nullptr) != cudaSuccess) {
    state.SkipWithError("warmup synchronize failed");
  }

#ifdef SYCLFFT_VERIFY_BENCHMARK
  using info = typename decltype(cu_state)::type_info;
  verify_dft<info::plan_type>(in, out, lengths, batch);
#endif  // SYCLFFT_VERIFY_BENCHMARK

  // benchmark
  for (auto _ : state) {
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    cufft_exec<typename decltype(cu_state)::type_info>(plan, in, out);
    cudaStreamSynchronize(nullptr);
    auto end = clock::now();

    double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    state.SetIterationTime(seconds);
    state.counters["flops"] = ops_est / seconds;
    state.counters["throughput"] = bytes_transfered / seconds;
  }
}

template <std::size_t runs, typename forward_type>
static void cufft_oop_average_host_time(benchmark::State& state, std::vector<int> lengths, int batch) noexcept {
  // setup state
  cufft_state<forward_type> cu_state(state, lengths, batch);

  // remove all the extra guff stored in the state
  auto plan = cu_state.plan.handle.value();
  auto in = cu_state.in.get();
  auto out = cu_state.out.get();

  // ops estimate for flops
  const auto fft_size = get_forward_fft_size(lengths);
  const auto ops_est = cooley_tukey_ops_estimate(fft_size, batch);
  using forward_info = typename forward_type_info<forward_type>;
  const int out_size = get_backward_fft_size<forward_info::plan_type>(lengths);
  const auto bytes_transfered =
      global_mem_transactions<typename forward_info::device_forward_type, typename forward_info::device_backward_type>(
          batch, fft_size, out_size);

  // warmup
  if (cufft_exec<typename decltype(cu_state)::type_info>(plan, in, out) != CUFFT_SUCCESS) {
    state.SkipWithError("warmup exec failed");
  }
  if (cudaStreamSynchronize(nullptr) != cudaSuccess) {
    state.SkipWithError("warmup synchronize failed");
  }

#ifdef SYCLFFT_VERIFY_BENCHMARK
  using info = typename decltype(cu_state)::type_info;
  verify_dft<info::plan_type>(in, out, lengths, batch);
#endif  // SYCLFFT_VERIFY_BENCHMARK

  // benchmark
  for (auto _ : state) {
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    for (std::size_t r = 0; r != runs; r += 1) {
      cufft_exec<typename decltype(cu_state)::type_info>(plan, in, out);
    }
    cudaStreamSynchronize(nullptr);
    auto end = clock::now();

    double seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() / static_cast<double>(runs);
    state.SetIterationTime(seconds);
    state.counters["flops"] = ops_est / seconds;
    state.counters["throughput"] = bytes_transfered / seconds;
  }
}

template <typename forward_type>
static void cufft_oop_device_time(benchmark::State& state, std::vector<int> lengths, int batch) noexcept {
  // setup state
  cufft_state<forward_type> cu_state(state, lengths, batch);

  // remove all the extra guff stored in the state
  auto plan = cu_state.plan.handle.value();
  auto in = cu_state.in.get();
  auto out = cu_state.out.get();

  // ops estimate for flops
  const auto fft_size = get_forward_fft_size(lengths);
  const auto ops_est = cooley_tukey_ops_estimate(fft_size, batch);
  using forward_info = typename forward_type_info<forward_type>;
  const int out_size = get_backward_fft_size<forward_info::plan_type>(lengths);
  const auto bytes_transfered =
      global_mem_transactions<typename forward_info::device_forward_type, typename forward_info::device_backward_type>(
          batch, fft_size, out_size);

  // warmup
  if (cufft_exec<typename decltype(cu_state)::type_info>(plan, in, out) != CUFFT_SUCCESS) {
    state.SkipWithError("warmup exec failed");
  }
  if (cudaStreamSynchronize(nullptr) != cudaSuccess) {
    state.SkipWithError("warmup synchronize failed");
  }

#ifdef SYCLFFT_VERIFY_BENCHMARK
  using info = typename decltype(cu_state)::type_info;
  verify_dft<info::plan_type>(in, out, lengths, batch);
#endif  // SYCLFFT_VERIFY_BENCHMARK

  cudaEvent_t before;
  cudaEvent_t after;

  if (cudaEventCreate(&before) != cudaSuccess || cudaEventCreate(&after) != cudaSuccess) {
    state.SkipWithError("event creation failed");
  }

  // benchmark
  for (auto _ : state) {
    auto before_res = cudaEventRecord(before);
    auto exec_res = cufft_exec<typename decltype(cu_state)::type_info>(plan, in, out);
    auto after_res = cudaEventRecord(after);
    auto sync_res = cudaEventSynchronize(after);
    if (before_res != cudaSuccess || exec_res != CUFFT_SUCCESS || after_res != cudaSuccess || sync_res != cudaSuccess) {
      state.SkipWithError("benchmark run failed");
    }
    float ms;
    if (cudaEventElapsedTime(&ms, before, after) != cudaSuccess) {
      state.SkipWithError("cudaEventElapsedTime failed");
    }
    double seconds = ms / 1000.0;
    state.SetIterationTime(seconds);
    state.counters["flops"] = ops_est / seconds;
    state.counters["throughput"] = bytes_transfered / seconds;
  }

  if (cudaEventDestroy(before) != cudaSuccess || cudaEventDestroy(after) != cudaSuccess) {
    state.SkipWithError("event destroy failed");
  }
}

// Helper functions for GBench
template <typename... Args>
void cufft_oop_real_time_complex_float(Args&&... args) {
  cufft_oop_real_time<std::complex<float>>(std::forward<Args>(args)...);
}

template <typename... Args>
void cufft_oop_real_time_float(Args&&... args) {
  cufft_oop_real_time<float>(std::forward<Args>(args)...);
}

template <typename... Args>
void cufft_oop_average_host_time_complex_float(Args&&... args) {
  cufft_oop_average_host_time<runs_to_average, std::complex<float>>(std::forward<Args>(args)...);
}

template <typename... Args>
void cufft_oop_average_host_time_float(Args&&... args) {
  cufft_oop_average_host_time<runs_to_average, float>(std::forward<Args>(args)...);
}

template <typename... Args>
void cufft_oop_device_time_complex_float(Args&&... args) {
  cufft_oop_device_time<std::complex<float>>(std::forward<Args>(args)...);
}

template <typename... Args>
void cufft_oop_device_time_float(Args&&... args) {
  cufft_oop_device_time<float>(std::forward<Args>(args)...);
}

#define BENCH_COMPLEX_FLOAT(...)                                                              \
  BENCHMARK_CAPTURE(cufft_oop_real_time_complex_float, __VA_ARGS__)->UseManualTime();         \
  BENCHMARK_CAPTURE(cufft_oop_average_host_time_complex_float, __VA_ARGS__)->UseManualTime(); \
  BENCHMARK_CAPTURE(cufft_oop_device_time_complex_float, __VA_ARGS__)->UseManualTime();

#define BENCH_SINGLE_FLOAT(...)                                                       \
  BENCHMARK_CAPTURE(cufft_oop_real_time_float, __VA_ARGS__)->UseManualTime();         \
  BENCHMARK_CAPTURE(cufft_oop_average_host_time_float, __VA_ARGS__)->UseManualTime(); \
  BENCHMARK_CAPTURE(cufft_oop_device_time_float, __VA_ARGS__)->UseManualTime();

INSTANTIATE_REFERENCE_BENCHMARK_SET(BENCH_COMPLEX_FLOAT, BENCH_SINGLE_FLOAT);

BENCHMARK_MAIN();
