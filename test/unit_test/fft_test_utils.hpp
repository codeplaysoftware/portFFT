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

#include "instantiate_fft_tests.hpp"
#include "number_generators.hpp"
#include "reference_dft.hpp"
#include "utils.hpp"
#include <sycl_fft.hpp>

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

using namespace sycl_fft;

using param_tuple = std::tuple<std::size_t, std::size_t>;

struct test_params {
  std::size_t batch;
  std::size_t length;
  test_params(param_tuple params) : batch(std::get<0>(params)), length(std::get<1>(params)) {}
};

void operator<<(std::ostream& stream, const test_params& params) {
  stream << "Batch = " << params.batch << ", Length = " << params.length;
}

template <typename TypeIn, typename TypeOut>
void transpose(TypeIn in, TypeOut& out, std::size_t FFT_size, std::size_t batch_size) {
  for (std::size_t j = 0; j < batch_size; j++) {
    for (std::size_t i = 0; i < FFT_size; i++) {
      out[i + j * FFT_size] = in[j + i * batch_size];
    }
  }
}

// test for out-of-place and in-place ffts.
template <typename FType, placement Place, direction Dir, bool TransposeIn = false>
void check_fft_usm(test_params& params, sycl::queue& queue) {
  ASSERT_TRUE(params.length > 0);
  if (TransposeIn && params.length > 13) {  // while we only support TransposeIn for workitem sizes
    GTEST_SKIP();
  }
  auto num_elements = params.batch * params.length;
  std::vector<std::complex<FType>> host_input(num_elements);
  std::vector<std::complex<FType>> host_reference_output(num_elements);
  std::vector<std::complex<FType>> buffer(num_elements);

  auto device_input = sycl::malloc_device<std::complex<FType>>(num_elements, queue);
  std::complex<FType>* device_output = nullptr;
  if (Place == placement::OUT_OF_PLACE) {
    device_output = sycl::malloc_device<std::complex<FType>>(num_elements, queue);
  }
  populate_with_random(host_input, FType(-1.0), FType(1.0));

  auto copy_event = queue.copy(host_input.data(), device_input, num_elements);

  descriptor<FType, domain::COMPLEX> desc{{params.length}};
  desc.number_of_transforms = params.batch;
  if constexpr (TransposeIn) {
    if constexpr (Dir == direction::FORWARD) {
      desc.forward_strides = {static_cast<std::size_t>(params.batch)};
      desc.forward_distance = 1;
    } else {
      desc.backward_strides = {static_cast<std::size_t>(params.batch)};
      desc.backward_distance = 1;
    }
  }
  auto committed_descriptor = desc.commit(queue);

  auto fft_event = [&]() {
    if constexpr (Place == placement::OUT_OF_PLACE) {
      if constexpr (Dir == direction::FORWARD) {
        return committed_descriptor.compute_forward(device_input, device_output, {copy_event});
      } else {
        return committed_descriptor.compute_backward(device_input, device_output, {copy_event});
      }
    } else {
      if constexpr (Dir == direction::FORWARD) {
        return committed_descriptor.compute_forward(device_input, {copy_event});
      } else {
        return committed_descriptor.compute_backward(device_input, {copy_event});
      }
    }
  }();

  double scaling_factor = Dir == direction::FORWARD ? desc.forward_scale : desc.backward_scale;
  std::vector<std::complex<FType>> host_input_transposed;
  if constexpr (TransposeIn) {
    host_input_transposed = std::vector<std::complex<FType>>(num_elements);
    transpose(host_input, host_input_transposed, params.length, params.batch);
  }
  for (std::size_t i = 0; i < params.batch; i++) {
    const auto offset = i * params.length;
    std::complex<FType>* input_for_reference =
        (TransposeIn ? host_input_transposed.data() : host_input.data()) + offset;
    reference_dft<Dir>(input_for_reference, host_reference_output.data() + offset, {params.length}, scaling_factor);
  }

  queue.copy(Place == placement::OUT_OF_PLACE ? device_output : device_input, buffer.data(), num_elements,
             {fft_event});
  queue.wait();
  compare_arrays(host_reference_output, buffer, 1e-3);
  sycl::free(device_input, queue);
  if (Place == placement::OUT_OF_PLACE) {
    sycl::free(device_output, queue);
  }
}

template <typename FType, placement Place, direction Dir, bool TransposeIn = false>
void check_fft_buffer(test_params& params, sycl::queue& queue) {
  ASSERT_TRUE(params.length > 0);
  if (TransposeIn && params.length > 13) {  // while we only support TransposeIn for workitem sizes
    GTEST_SKIP();
  }
  auto num_elements = params.batch * params.length;
  std::vector<std::complex<FType>> host_input(num_elements);
  std::vector<std::complex<FType>> host_reference_output(num_elements);
  std::vector<std::complex<FType>> buffer(num_elements);

  populate_with_random(host_input, FType(-1.0), FType(1.0));

  descriptor<FType, domain::COMPLEX> desc{{static_cast<unsigned long>(params.length)}};
  desc.number_of_transforms = params.batch;
  if constexpr (TransposeIn) {
    if constexpr (Dir == direction::FORWARD) {
      desc.forward_strides = {static_cast<std::size_t>(params.batch)};
      desc.forward_distance = 1;
    } else {
      desc.backward_strides = {static_cast<std::size_t>(params.batch)};
      desc.backward_distance = 1;
    }
  }
  auto committed_descriptor = desc.commit(queue);
  double scaling_factor = Dir == direction::FORWARD ? desc.forward_scale : desc.backward_scale;

  std::vector<std::complex<FType>> host_input_transposed;
  if constexpr (TransposeIn) {
    host_input_transposed = std::vector<std::complex<FType>>(num_elements);
    transpose(host_input, host_input_transposed, params.length, params.batch);
  }
  for (std::size_t i = 0; i < params.batch; i++) {
    const auto offset = i * params.length;
    std::complex<FType>* input_for_reference =
        (TransposeIn ? host_input_transposed.data() : host_input.data()) + offset;
    reference_dft<Dir>(input_for_reference, host_reference_output.data() + offset, {params.length}, scaling_factor);
  }

  {
    sycl::buffer<std::complex<FType>, 1> output_buffer(nullptr, 0);
    sycl::buffer<std::complex<FType>, 1> input_buffer(host_input.data(), num_elements);
    if (Place == placement::OUT_OF_PLACE) {
      output_buffer = sycl::buffer<std::complex<FType>, 1>(buffer.data(), num_elements);
    }

    if constexpr (Place == placement::OUT_OF_PLACE) {
      if constexpr (Dir == direction::FORWARD) {
        committed_descriptor.compute_forward(input_buffer, output_buffer);
      } else {
        committed_descriptor.compute_backward(input_buffer, output_buffer);
      }
    } else {
      if constexpr (Dir == direction::FORWARD) {
        committed_descriptor.compute_forward(input_buffer);
      } else {
        committed_descriptor.compute_backward(input_buffer);
      }
    }
    queue.wait();
  }
  compare_arrays(host_reference_output, Place == placement::IN_PLACE ? host_input : buffer, 1e-3);
}

#endif
