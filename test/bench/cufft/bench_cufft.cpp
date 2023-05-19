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

template <typename forward_type, typename custate>
void cufft_verify(const custate& cu_state, const std::vector<int>& lengths, const int batch) {
  using info = typename custate::type_info;

  auto fwd_copy = std::make_unique<forward_type[]>(cu_state.fwd_per_transform * batch);
  auto bwd_copy = std::make_unique<typename info::backward_type[]>(cu_state.bwd_per_transform * batch);
  cudaMemcpy(fwd_copy.get(), cu_state.in.get(), cu_state.fwd_per_transform * batch * sizeof(forward_type),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(bwd_copy.get(), cu_state.out.get(),
             cu_state.bwd_per_transform * batch * sizeof(typename info::backward_type), cudaMemcpyDeviceToHost);
  verify_dft<forward_type, typename info::backward_type>(fwd_copy.get(), bwd_copy.get(), lengths, batch, 1.0);
}

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
  int fwd_per_transform;
  int bwd_per_transform;

  cufft_state(benchmark::State& state, std::vector<int> lengths, const int batch)
      : test_state(state),
        plan(state, {}),
        in(nullptr, cuda_freer<typename type_info::device_forward_type>{state}),
        out(nullptr, cuda_freer<typename type_info::device_backward_type>{state}),
        fwd_per_transform(get_fwd_per_transform(lengths)),
        bwd_per_transform(get_bwd_per_transform<forward_type>(lengths)) {
    if (lengths.empty()) {
      test_state.SkipWithError("invalid configuration");
    }
    // nullptr inembed and onembed is equivalent to giving the lengths for both
    int *inembed = nullptr, *onembed = nullptr;
    int istride = 1;
    int ostride = 1;
    int idist = fwd_per_transform;
    int odist = bwd_per_transform;
    cufftHandle plan_tmp;
    auto res = cufftPlanMany(&plan_tmp, lengths.size(), lengths.data(), inembed, istride, idist, onembed, ostride,
                             odist, type_info::plan_type, batch);
    if (res == CUFFT_SUCCESS) {
      plan.handle = plan_tmp;
    } else {
      test_state.SkipWithError("plan creation failed");
    }

    typename type_info::device_forward_type* in_tmp;
    // TODO overallocing in the REAL-COMPLEX case
    if (cudaMalloc(&in_tmp, sizeof(forward_type) * fwd_per_transform * batch) == cudaSuccess) {
      in.reset(in_tmp);
    } else {
      test_state.SkipWithError("in allocation failed");
    }

    typename type_info::device_backward_type* out_tmp;
    if (cudaMalloc(&out_tmp, sizeof(typename type_info::backward_type) * bwd_per_transform * batch) == cudaSuccess) {
      out.reset(out_tmp);
    } else {
      test_state.SkipWithError("out allocation failed");
    }
#ifdef SYCLFFT_VERIFY_BENCHMARK
    populate_with_random(reinterpret_cast<typename scalar_data_type<type_info::plan_type>::type*>(in.get()),
                         fwd_per_transform * batch);
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
static void cufft_oop_average_host_time(benchmark::State& state, std::vector<int> lengths, int batch,
                                        std::size_t runs) noexcept {
  // setup state
  cufft_state<forward_type> cu_state(state, lengths, batch);
  using info = typename decltype(cu_state)::type_info;

  // remove all the extra guff stored in the state
  auto plan = cu_state.plan.handle.value();
  auto in = cu_state.in.get();
  auto out = cu_state.out.get();

  // ops estimate for flops
  const auto ops_est = cooley_tukey_ops_estimate(cu_state.fwd_per_transform, batch);
  const auto bytes_transfered =
      global_mem_transactions<typename info::device_forward_type, typename info::device_backward_type>(
          batch, cu_state.fwd_per_transform, cu_state.bwd_per_transform);

  // warmup
  if (cufft_exec<info>(plan, in, out) != CUFFT_SUCCESS) {
    state.SkipWithError("warmup exec failed");
  }
  if (cudaStreamSynchronize(nullptr) != cudaSuccess) {
    state.SkipWithError("warmup synchronize failed");
  }

#ifdef SYCLFFT_VERIFY_BENCHMARK
  cufft_verify<forward_type>(cu_state, lengths, batch);
#endif  // SYCLFFT_VERIFY_BENCHMARK

  // benchmark
  for (auto _ : state) {
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    for (std::size_t r = 0; r != runs; r += 1) {
      cufft_exec<info>(plan, in, out);
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
  using info = typename decltype(cu_state)::type_info;

  // remove all the extra guff stored in the state
  auto plan = cu_state.plan.handle.value();
  auto in = cu_state.in.get();
  auto out = cu_state.out.get();

  // ops estimate for flops
  const auto ops_est = cooley_tukey_ops_estimate(cu_state.fwd_per_transform, batch);
  const auto bytes_transfered =
      global_mem_transactions<typename info::device_forward_type, typename info::device_backward_type>(
          batch, cu_state.fwd_per_transform, cu_state.bwd_per_transform);

  // warmup
  if (cufft_exec<info>(plan, in, out) != CUFFT_SUCCESS) {
    state.SkipWithError("warmup exec failed");
  }
  if (cudaStreamSynchronize(nullptr) != cudaSuccess) {
    state.SkipWithError("warmup synchronize failed");
  }

#ifdef SYCLFFT_VERIFY_BENCHMARK
  cufft_verify<forward_type>(cu_state, lengths, batch);
#endif  // SYCLFFT_VERIFY_BENCHMARK

  cudaEvent_t before;
  cudaEvent_t after;

  if (cudaEventCreate(&before) != cudaSuccess || cudaEventCreate(&after) != cudaSuccess) {
    state.SkipWithError("event creation failed");
  }

  // benchmark
  for (auto _ : state) {
    auto before_res = cudaEventRecord(before);
    auto exec_res = cufft_exec<info>(plan, in, out);
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
void cufft_oop_average_host_time_complex_float(Args&&... args) {
  cufft_oop_average_host_time<std::complex<float>>(std::forward<Args>(args)..., runs_to_average);
}

template <typename... Args>
void cufft_oop_average_host_time_float(Args&&... args) {
  cufft_oop_average_host_time<float>(std::forward<Args>(args)..., runs_to_average);
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
  BENCHMARK_CAPTURE(cufft_oop_average_host_time_complex_float, __VA_ARGS__)->UseManualTime(); \
  BENCHMARK_CAPTURE(cufft_oop_device_time_complex_float, __VA_ARGS__)->UseManualTime();

#define BENCH_SINGLE_FLOAT(...)                                                       \
  BENCHMARK_CAPTURE(cufft_oop_average_host_time_float, __VA_ARGS__)->UseManualTime(); \
  BENCHMARK_CAPTURE(cufft_oop_device_time_float, __VA_ARGS__)->UseManualTime();

INSTANTIATE_REFERENCE_BENCHMARK_SET(BENCH_COMPLEX_FLOAT, BENCH_SINGLE_FLOAT);

BENCHMARK_MAIN();
