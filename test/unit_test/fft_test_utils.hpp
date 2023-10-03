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

#ifndef PORTFFT_UNIT_TEST_FFT_TEST_UTILS
#define PORTFFT_UNIT_TEST_FFT_TEST_UTILS

#include "instantiate_fft_tests.hpp"
#include "utils.hpp"
#include <portfft.hpp>
#include <reference_data_wrangler.hpp>

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

using namespace portfft;

using param_tuple = std::tuple<std::size_t, std::size_t>;

struct test_params {
  std::size_t batch;
  std::size_t length;
  test_params(param_tuple params) : batch(std::get<0>(params)), length(std::get<1>(params)) {}
};

void operator<<(std::ostream& stream, const test_params& params) {
  stream << "Batch = " << params.batch << ", Length = " << params.length;
}

/**
 * Runs Out of place transpose
 *
 * @tparam TypeIn Input Type
 * @tparam TypeOut Output Type
 * @param in input pointer
 * @param out output pointer
 * @param FFT_size innermost dimension of the input
 * @param batch_size innermost dimension of the output
 */
template <typename TypeIn, typename TypeOut>
void transpose(TypeIn in, TypeOut& out, std::size_t FFT_size, std::size_t batch_size) {
  for (std::size_t j = 0; j < batch_size; j++) {
    for (std::size_t i = 0; i < FFT_size; i++) {
      out[i + j * FFT_size] = in[j + i * batch_size];
    }
  }
}

/**
 * Runs USM FFT Test for the given length, batch
 *
 * @tparam FType Scalar type Float / Double
 * @tparam in-place or out-of-place
 * @tparam Dir Direction of the transform
 * @tparam LayoutIn Input Layout to the FFT
 * @tparam LayoutOut Output Layout of the obtained FFT
 * @param params Param struct containing length, batch
 * @param queue Associated queue
 */
template <typename FType, placement Place, direction Dir, detail::layout LayoutIn, detail::layout LayoutOut>
void check_fft_usm(test_params& params, sycl::queue& queue) {
  ASSERT_TRUE(params.length > 0);
  {
    std::vector<std::size_t> instantiated_sizes{PORTFFT_COOLEY_TUKEY_OPTIMIZED_SIZES};
    if (!std::count(instantiated_sizes.cbegin(), instantiated_sizes.cend(), params.length)) {
      GTEST_SKIP() << "Test skipped as test size not present in optimized size list";
    }
  }
  constexpr bool IsBatchInterleaved = LayoutIn == detail::layout::BATCH_INTERLEAVED;
  auto num_elements = params.batch * params.length;
  std::vector<std::complex<FType>> host_input(num_elements);
  std::vector<std::complex<FType>> host_input_transposed;
  std::vector<std::complex<FType>> host_reference_output(num_elements);
  std::vector<std::complex<FType>> buffer(num_elements);

  auto device_input = sycl::malloc_device<std::complex<FType>>(num_elements, queue);
  std::complex<FType>* device_output = nullptr;
  if (Place == placement::OUT_OF_PLACE) {
    device_output = sycl::malloc_device<std::complex<FType>>(num_elements, queue);
  }

  descriptor<FType, domain::COMPLEX> desc{{params.length}};
  desc.number_of_transforms = params.batch;
  if constexpr (Dir == direction::FORWARD) {
    if constexpr (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
      desc.forward_strides = {static_cast<std::size_t>(params.batch)};
      desc.forward_distance = 1;
    }
    if constexpr (LayoutOut == detail::layout::BATCH_INTERLEAVED) {
      desc.backward_strides = {static_cast<std::size_t>(params.batch)};
      desc.backward_distance = 1;
    }
  }
  if constexpr (Dir == direction::BACKWARD) {
    if constexpr (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
      desc.backward_strides = {static_cast<std::size_t>(params.batch)};
      desc.backward_distance = 1;
    }
    if constexpr (LayoutOut == detail::layout::BATCH_INTERLEAVED) {
      desc.forward_strides = {static_cast<std::size_t>(params.batch)};
      desc.forward_distance = 1;
    }
  }

  auto committed_descriptor = desc.commit(queue);

  auto [forward_data, backward_data] = gen_fourier_data(desc);

  if constexpr (Dir == portfft::direction::FORWARD) {
    host_input = forward_data;
    host_reference_output = backward_data;
  } else {
    host_input = backward_data;
    host_reference_output = forward_data;
  }
  if constexpr (IsBatchInterleaved) {
    host_input_transposed = std::vector<std::complex<FType>>(num_elements);
    transpose(host_input, host_input_transposed, params.batch, params.length);
  }

  auto copy_event =
      queue.copy(IsBatchInterleaved ? host_input_transposed.data() : host_input.data(), device_input, num_elements);

  sycl::event fft_event = [&]() {
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
  queue.copy(Place == placement::OUT_OF_PLACE ? device_output : device_input, buffer.data(), num_elements, {fft_event});
  queue.wait();
  verify_dft(desc, host_reference_output, buffer, Dir, LayoutOut, 1e-3);

  sycl::free(device_input, queue);
  if (Place == placement::OUT_OF_PLACE) {
    sycl::free(device_output, queue);
  }
}

/**
 * Runs Buffer FFT Test for the given length, batch
 *
 * @tparam FType Scalar type Float / Double
 * @tparam Place In place or Out of place
 * @tparam Dir Direction of the transform
 * @tparam LayoutIn Input Layout to the FFT
 * @tparam LayoutOut Output Layout of the obtained FFT
 * @param params Param struct containing length, batch
 * @param queue Associated queue
 */
template <typename FType, placement Place, direction Dir, detail::layout LayoutIn, detail::layout LayoutOut>
void check_fft_buffer(test_params& params, sycl::queue& queue) {
  ASSERT_TRUE(params.length > 0);
  {
    std::vector<std::size_t> instantiated_sizes{PORTFFT_COOLEY_TUKEY_OPTIMIZED_SIZES};
    if (!std::count(instantiated_sizes.cbegin(), instantiated_sizes.cend(), params.length)) {
      GTEST_SKIP() << "Test skipped as test size not present in optimized size list";
    }
  }
  constexpr bool IsBatchInterleaved = LayoutIn == detail::layout::BATCH_INTERLEAVED;
  auto num_elements = params.batch * params.length;
  std::vector<std::complex<FType>> host_input_raw(num_elements);
  std::vector<std::complex<FType>> host_reference_output(num_elements);
  std::vector<std::complex<FType>> buffer(num_elements);

  descriptor<FType, domain::COMPLEX> desc{{static_cast<unsigned long>(params.length)}};
  desc.number_of_transforms = params.batch;

  if constexpr (Dir == direction::FORWARD) {
    if constexpr (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
      desc.forward_strides = {static_cast<std::size_t>(params.batch)};
      desc.forward_distance = 1;
    }
    if constexpr (LayoutOut == detail::layout::BATCH_INTERLEAVED) {
      desc.backward_distance = 1;
      desc.backward_strides = {static_cast<std::size_t>(params.batch)};
    }
  }
  if constexpr (Dir == direction::BACKWARD) {
    if constexpr (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
      desc.backward_distance = 1;
      desc.backward_strides = {static_cast<std::size_t>(params.batch)};
    }
    if constexpr (LayoutOut == detail::layout::BATCH_INTERLEAVED) {
      desc.forward_strides = {static_cast<std::size_t>(params.batch)};
      desc.forward_distance = 1;
    }
  }

  auto committed_descriptor = desc.commit(queue);

  auto [forward_data, backward_data] = gen_fourier_data(desc);
  if constexpr (Dir == portfft::direction::FORWARD) {
    host_input_raw = forward_data;
    host_reference_output = backward_data;
  } else {
    host_input_raw = backward_data;
    host_reference_output = forward_data;
  }
  std::vector<std::complex<FType>> host_input(host_input_raw);
  if constexpr (IsBatchInterleaved) {
    transpose(host_input_raw, host_input, params.batch, params.length);
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
  }
  verify_dft(desc, host_reference_output, Place == placement::IN_PLACE ? host_input : buffer, Dir, LayoutOut, 1e-3);
}

#endif
