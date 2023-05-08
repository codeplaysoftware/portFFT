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

// Intel's closed-source OneMKL library header.
#include <oneapi/mkl/dfti.hpp>
#include <oneapi/mkl/exceptions.hpp>

#include "bench_onemkl_utils.hpp"

// Define backend specific methods

template <typename forward_type>
void onemkl_state<forward_type>::set_out_of_place() {
  desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
}

template <typename forward_type>
sycl::event onemkl_state<forward_type>::compute(const std::vector<sycl::event>& deps) {
  return compute_forward(desc, in_dev, out_dev, deps);
}
