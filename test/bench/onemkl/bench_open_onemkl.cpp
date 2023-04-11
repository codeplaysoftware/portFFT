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
 *  Building with open-source OneMKL:
 *  Set SYCLFFT_INTEL_OPEN_ONEMKL_BENCHMARK_BACKEND=<backend>.
 *
 **************************************************************************/

// Intel's open-source OneMKL library header.
#include <oneapi/mkl.hpp>

#include "bench_onemkl_utils.hpp"

// Define backend specific methods

template <oneapi::mkl::dft::precision prec, oneapi::mkl::dft::domain domain>
void onemkl_state<prec, domain>::set_out_of_place() {
  desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, oneapi::mkl::dft::config_value::NOT_INPLACE);
}

template <oneapi::mkl::dft::precision prec, oneapi::mkl::dft::domain domain>
sycl::event onemkl_state<prec, domain>::compute() {
  return oneapi::mkl::dft::compute_forward<descriptor_t, complex_t, complex_t>(desc, in_dev, out_dev);
}
