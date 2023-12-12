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

#include <portfft/traits.hpp>

#include "launch_bench.hpp"
#include "utils/sycl_utils.hpp"

template <typename T>
void bench_dft(sycl::queue q, sycl::queue profiling_q, const std::string& suffix,
               const std::vector<std::size_t>& lengths, std::size_t batch) {
  using ftype = typename portfft::get_real<T>::type;
  constexpr portfft::domain domain = portfft::get_domain<T>::value;

  portfft::descriptor<ftype, domain> desc(lengths);
  desc.number_of_transforms = batch;

  register_host_device_benchmark(suffix, q, profiling_q, desc);
}

int main(int argc, char** argv) {
  using ftype = float;
  benchmark::SetDefaultTimeUnit(benchmark::kMillisecond);
  benchmark::Initialize(&argc, argv);

  sycl::queue q(sycl::gpu_selector_v);
  sycl::queue profiling_q(sycl::gpu_selector_v, {sycl::property::queue::enable_profiling()});
  print_device(q);

  // Benchmark configurations must match with the ones in test/bench/utils/reference_dft_set.hpp
  // Configurations are progressively added as portFFT supports more of them.
  bench_dft<std::complex<ftype>>(q, profiling_q, "small_1d", {16}, 8 * 1024 * 1024);
  bench_dft<std::complex<ftype>>(q, profiling_q, "medium_small_1d", {256}, 512 * 1024);
  bench_dft<std::complex<ftype>>(q, profiling_q, "medium_large_1d", {4096}, 32 * 1024);
  bench_dft<std::complex<ftype>>(q, profiling_q, "large_1d", {65536}, 2048);

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
