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
 *  Codeplay's SYCL-FFT
 *
 **************************************************************************/

#ifndef SYCL_FFT_UNIT_TEST_FFT_TEST_UTILS
#define SYCL_FFT_UNIT_TEST_FFT_TEST_UTILS

#include "descriptor.hpp"
#include "enums.hpp"
#include "instantiate_fft_tests.hpp"
#include "number_generators.hpp"
#include "reference_dft.hpp"
#include "utils.hpp"

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

using namespace sycl_fft;

// test for out-of-place and in-place ffts.
template <typename ftype, placement test_type, direction dir>
void check_fft_usm(test_params& params, sycl::queue& queue) {
  ASSERT_TRUE(params.length > 0);
  auto num_elements = params.batch * params.length;
  std::vector<std::complex<ftype>> host_input(num_elements);
  std::vector<std::complex<ftype>> host_reference_output(num_elements);
  std::vector<std::complex<ftype>> buffer(num_elements);

  auto device_input = sycl::malloc_device<std::complex<ftype>>(num_elements, queue);
  std::complex<ftype>* device_output = nullptr;
  if (test_type == placement::OUT_OF_PLACE) {
    device_output = sycl::malloc_device<std::complex<ftype>>(num_elements, queue);
  }
  populate_with_random(host_input, ftype(-1.0), ftype(1.0));

  auto copy_event = queue.copy(host_input.data(), device_input, num_elements);

  descriptor<ftype, domain::COMPLEX> desc{{params.length}};
  desc.number_of_transforms = params.batch;
  auto committed_descriptor = desc.commit(queue);

  auto fft_event = [&]() {
    if constexpr (test_type == placement::OUT_OF_PLACE) {
      if constexpr (dir == direction::FORWARD) {
        return committed_descriptor.compute_forward(device_input, device_output, {copy_event});
      } else {
        return committed_descriptor.compute_backward(device_input, device_output, {copy_event});
      }
    } else {
      if constexpr (dir == direction::FORWARD) {
        return committed_descriptor.compute_forward(device_input, {copy_event});
      } else {
        return committed_descriptor.compute_backward(device_input, {copy_event});
      }
    }
  }();

  double scaling_factor = dir == direction::FORWARD ? desc.forward_scale : desc.backward_scale;
  for (std::size_t i = 0; i < params.batch; i++) {
    const auto offset = i * params.length;
    reference_dft<dir>(host_input.data() + offset, host_reference_output.data() + offset, {params.length},
                       scaling_factor);
  }

  queue.copy(test_type == placement::OUT_OF_PLACE ? device_output : device_input, buffer.data(), num_elements,
             {fft_event});
  queue.wait();
  compare_arrays(host_reference_output, buffer, 1e-5);
  sycl::free(device_input, queue);
  if (test_type == placement::OUT_OF_PLACE) {
    sycl::free(device_output, queue);
  }
}

template <typename ftype, placement test_type, direction dir>
void check_fft_buffer(test_params& params, sycl::queue& queue) {
  ASSERT_TRUE(params.length > 0);
  auto num_elements = params.batch * params.length;
  std::vector<std::complex<ftype>> host_input(num_elements);
  std::vector<std::complex<ftype>> host_reference_output(num_elements);
  std::vector<std::complex<ftype>> buffer(num_elements);

  populate_with_random(host_input, ftype(-1.0), ftype(1.0));
  {
    sycl::buffer<std::complex<ftype>, 1> output_buffer(nullptr, 0);
    sycl::buffer<std::complex<ftype>, 1> input_buffer(host_input.data(), num_elements);
    if (test_type == placement::OUT_OF_PLACE) {
      output_buffer = sycl::buffer<std::complex<ftype>, 1>(buffer.data(), num_elements);
    }

    descriptor<ftype, domain::COMPLEX> desc{{params.length}};
    desc.number_of_transforms = params.batch;
    auto committed_descriptor = desc.commit(queue);
    double scaling_factor = dir == direction::FORWARD ? desc.forward_scale : desc.backward_scale;
    for (std::size_t i = 0; i < params.batch; i++) {
      const auto offset = i * params.length;
      reference_dft<dir>(host_input.data() + offset, host_reference_output.data() + offset, {params.length},
                         scaling_factor);
    }

    if constexpr (test_type == placement::OUT_OF_PLACE) {
      if constexpr (dir == direction::FORWARD) {
        committed_descriptor.compute_forward(input_buffer, output_buffer);
      } else {
        committed_descriptor.compute_backward(input_buffer, output_buffer);
      }
    } else {
      if constexpr (dir == direction::FORWARD) {
        committed_descriptor.compute_forward(input_buffer);
      } else {
        committed_descriptor.compute_backward(input_buffer);
      }
    }
    queue.wait();
  }
  compare_arrays(test_type == placement::IN_PLACE ? host_input : buffer, host_reference_output, 1e-5);
}

#endif
