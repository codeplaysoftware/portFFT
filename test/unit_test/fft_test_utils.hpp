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

template <typename TypeIn, typename TypeOut>
void transpose(TypeIn in, TypeOut& out, std::size_t FFT_size, std::size_t batch_size) {
  for (std::size_t j = 0; j < batch_size; j++) {
    for (std::size_t i = 0; i < FFT_size; i++) {
      out[i + j * FFT_size] = in[j + i * batch_size];
    }
  }
}

template <typename Scalar, portfft::domain Domain>
std::pair<std::optional<committed_descriptor<Scalar, Domain>>, std::string> get_committed_descriptor(
    descriptor<Scalar, Domain>& desc, sycl::queue& queue) {
  try {
    return std::make_pair(desc.commit(queue), "");
  } catch (std::runtime_error& e) {
    return std::make_pair(std::nullopt, e.what());
  }
}

template <typename Scalar, portfft::domain Domain, portfft::direction Dir, portfft::placement Place, typename T>
std::pair<std::optional<sycl::event>, std::string> run_compute(committed_descriptor<Scalar, Domain>& desc, T& input,
                                                               T& output, sycl::event& copy_event) {
  try {
    if constexpr (Place == placement::OUT_OF_PLACE) {
      if constexpr (Dir == direction::FORWARD) {
        return std::make_pair(desc.compute_forward(input, output, {copy_event}), "");
      } else {
        return std::make_pair(desc.compute_backward(input, output, {copy_event}), "");
      }
    } else {
      if constexpr (Dir == direction::FORWARD) {
        return std::make_pair(desc.compute_forward(input, {copy_event}), "");
      } else {
        return std::make_pair(desc.compute_backward(input, {copy_event}), "");
      }
    }
  } catch (portfft::inadequate_local_memory_error& error) {
    return std::make_pair(std::nullopt, error.what());
  }
}

template <typename Scalar, portfft::domain Domain, portfft::direction Dir, portfft::placement Place, typename T>
std::optional<std::string> run_compute(committed_descriptor<Scalar, Domain>& desc, T& input, T& output) {
  try {
    if constexpr (Place == placement::OUT_OF_PLACE) {
      if constexpr (Dir == direction::FORWARD) {
        desc.compute_forward(input, output);
      } else {
        desc.compute_backward(input, output);
      }
    } else {
      if constexpr (Dir == direction::FORWARD) {
        desc.compute_forward(input);
      } else {
        desc.compute_backward(input);
      }
    }
    return std::nullopt;
  } catch (portfft::inadequate_local_memory_error& error) {
    return error.what();
  }
}

// test for out-of-place and in-place ffts.
template <typename FType, placement Place, direction Dir, bool TransposeIn = false, bool TransposeOut = false>
void check_fft_usm(test_params& params, sycl::queue& queue) {
  ASSERT_TRUE(params.length > 0);
  {
    std::vector<std::size_t> instantiated_sizes{PORTFFT_COOLEY_TUKEY_OPTIMIZED_SIZES};
    if (!std::count(instantiated_sizes.cbegin(), instantiated_sizes.cend(), params.length)) {
      GTEST_SKIP() << "Test skipped as test size not present in optimized size list";
    }
  }
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
    if constexpr (TransposeIn) {
      desc.forward_strides = {static_cast<std::size_t>(params.batch)};
      desc.forward_distance = 1;
    }
    if constexpr (TransposeOut) {
      desc.backward_distance = 1;
      desc.backward_strides = {static_cast<std::size_t>(params.batch)};
    }
  }
  if constexpr (Dir == direction::BACKWARD) {
    if constexpr (TransposeIn) {
      desc.backward_distance = 1;
      desc.backward_strides = {static_cast<std::size_t>(params.batch)};
    }
    if constexpr (TransposeOut) {
      desc.forward_strides = {static_cast<std::size_t>(params.batch)};
      desc.forward_distance = 1;
    }
  }

  auto potential_committed_descriptor = get_committed_descriptor<FType, domain::COMPLEX>(desc, queue);
  if (!potential_committed_descriptor.first.has_value()) {
    GTEST_SKIP() << potential_committed_descriptor.second;
  }
  auto committed_descriptor = potential_committed_descriptor.first.value();

  auto verifSpec = get_matching_spec(verification_data, desc);
  if constexpr (Dir == portfft::direction::FORWARD) {
    host_input = verifSpec.template load_data_time(desc);
  } else {
    host_input = verifSpec.template load_data_fourier(desc);
  }
  if constexpr (TransposeIn) {
    host_input_transposed = std::vector<std::complex<FType>>(num_elements);
    transpose(host_input, host_input_transposed, params.batch, params.length);
  }

  auto copy_event =
      queue.copy(TransposeIn ? host_input_transposed.data() : host_input.data(), device_input, num_elements);

  auto potential_fft_event =
      run_compute<FType, domain::COMPLEX, Dir, Place>(committed_descriptor, device_input, device_output, copy_event);
  if (!potential_fft_event.first.has_value()) {
    GTEST_SKIP() << potential_fft_event.second;
  }
  sycl::event fft_event = potential_fft_event.first.value();
  queue.copy(Place == placement::OUT_OF_PLACE ? device_output : device_input, buffer.data(), num_elements, {fft_event});
  queue.wait();
  verifSpec.verify_dft(desc, buffer, Dir, TransposeOut, 1e-3);

  sycl::free(device_input, queue);
  if (Place == placement::OUT_OF_PLACE) {
    sycl::free(device_output, queue);
  }
}

template <typename FType, placement Place, direction Dir, bool TransposeIn = false, bool TransposeOut = false>
void check_fft_buffer(test_params& params, sycl::queue& queue) {
  ASSERT_TRUE(params.length > 0);
  {
    std::vector<std::size_t> instantiated_sizes{PORTFFT_COOLEY_TUKEY_OPTIMIZED_SIZES};
    if (!std::count(instantiated_sizes.cbegin(), instantiated_sizes.cend(), params.length)) {
      GTEST_SKIP() << "Test skipped as test size not present in optimized size list";
    }
  }
  auto num_elements = params.batch * params.length;
  std::vector<std::complex<FType>> host_input(num_elements);
  std::vector<std::complex<FType>> host_input_transposed;
  std::vector<std::complex<FType>> host_reference_output(num_elements);
  std::vector<std::complex<FType>> buffer(num_elements);

  descriptor<FType, domain::COMPLEX> desc{{static_cast<unsigned long>(params.length)}};
  desc.number_of_transforms = params.batch;

  if constexpr (Dir == direction::FORWARD) {
    if constexpr (TransposeIn) {
      desc.forward_strides = {static_cast<std::size_t>(params.batch)};
      desc.forward_distance = 1;
    }
    if constexpr (TransposeOut) {
      desc.backward_distance = 1;
      desc.backward_strides = {static_cast<std::size_t>(params.batch)};
    }
  }
  if constexpr (Dir == direction::BACKWARD) {
    if constexpr (TransposeIn) {
      desc.backward_distance = 1;
      desc.backward_strides = {static_cast<std::size_t>(params.batch)};
    }
    if constexpr (TransposeOut) {
      desc.forward_strides = {static_cast<std::size_t>(params.batch)};
      desc.forward_distance = 1;
    }
  }

  auto potential_committed_descriptor = get_committed_descriptor<FType, domain::COMPLEX>(desc, queue);
  if (!potential_committed_descriptor.first.has_value()) {
    GTEST_SKIP() << potential_committed_descriptor.second;
  }
  auto committed_descriptor = potential_committed_descriptor.first.value();

  auto verifSpec = get_matching_spec(verification_data, desc);
  if constexpr (Dir == portfft::direction::FORWARD) {
    host_input = verifSpec.template load_data_time(desc);
  } else {
    host_input = verifSpec.template load_data_fourier(desc);
  }
  if constexpr (TransposeIn) {
    host_input_transposed = std::vector<std::complex<FType>>(num_elements);
    transpose(host_input, host_input_transposed, params.batch, params.length);
  }

  {
    sycl::buffer<std::complex<FType>, 1> output_buffer(nullptr, 0);
    sycl::buffer<std::complex<FType>, 1> input_buffer(TransposeIn ? host_input_transposed.data() : host_input.data(),
                                                      num_elements);
    if (Place == placement::OUT_OF_PLACE) {
      output_buffer = sycl::buffer<std::complex<FType>, 1>(buffer.data(), num_elements);
    }

    auto possible_error =
        run_compute<FType, portfft::domain::COMPLEX, Dir, Place>(committed_descriptor, input_buffer, output_buffer);
    queue.wait_and_throw();
    if (possible_error.has_value()) {
      GTEST_SKIP() << possible_error.value();
    }
  }
  verifSpec.verify_dft(desc, Place == placement::IN_PLACE ? host_input : buffer, Dir, TransposeOut, 1e-3);
}

#endif
