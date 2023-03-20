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
#include "number_generators.hpp"
#include "utils.hpp"

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <tuple>

using namespace sycl_fft;
using param_tuple = std::tuple<int, int>;

struct test_params {
  int batch;
  int length;
  test_params(param_tuple params)
      : batch(std::get<0>(params)), length(std::get<1>(params)) {}
};

void operator<<(std::ostream& stream, const test_params& params) {
  stream << "Batch = " << params.batch << ", Length = " << params.length
         << '\n';
}

class FFTTest : public ::testing::TestWithParam<test_params> {
};  // batch, length

// test for out-of-place and in-place ffts.
template <typename ftype, placement test_type>
void check_fft_usm(test_params& params, sycl::queue& queue) {
  ASSERT_TRUE(params.length > 0);
  auto num_elements = params.batch * params.length;
  std::vector<std::complex<ftype>> host_input(num_elements);
  std::vector<std::complex<ftype>> host_reference_output(num_elements);
  std::vector<std::complex<ftype>> buffer(num_elements);

  auto device_input =
      sycl::malloc_device<std::complex<ftype>>(num_elements, queue);
  std::complex<ftype>* device_output = nullptr;
  if (test_type == placement::OUT_OF_PLACE) {
    device_output =
        sycl::malloc_device<std::complex<ftype>>(num_elements, queue);
  }
  populate_with_random(host_input, ftype(-1.0), ftype(1.0));

  auto copy_event = queue.copy(host_input.data(), device_input, num_elements);

  descriptor<ftype, domain::COMPLEX> desc{
      {static_cast<unsigned long>(params.length)}};
  desc.number_of_transforms = params.batch;
  auto commited_descriptor = desc.commit(queue);

  auto fft_event = test_type == placement::OUT_OF_PLACE ? 
                                commited_descriptor.compute_forward(device_input, device_output, {copy_event}) :
                                commited_descriptor.compute_forward(device_input, {copy_event});
  for (size_t i = 0; i < params.batch; i++)
    reference_forward_dft(host_input, host_reference_output, params.length,
                          i * params.length);
  queue.copy(
      test_type == placement::OUT_OF_PLACE ? device_output : device_input,
      buffer.data(), num_elements, {fft_event});
  queue.wait();
  compare_arrays(host_reference_output, buffer, 1e-5);
  sycl::free(device_input, queue);
  if (test_type == placement::OUT_OF_PLACE) {
    sycl::free(device_output, queue);
  }
}

template <typename ftype, placement test_type>
void check_fft_buffer(test_params& params, sycl::queue& queue) {
  ASSERT_TRUE(params.length > 0);
  auto num_elements = params.batch * params.length;
  std::vector<std::complex<ftype>> host_input(num_elements);
  std::vector<std::complex<ftype>> host_reference_output(num_elements);
  std::vector<std::complex<ftype>> buffer(num_elements);

  populate_with_random(host_input, ftype(-1.0), ftype(1.0));
  {
    sycl::buffer<std::complex<ftype>, 1> output_buffer(nullptr, 0);
    sycl::buffer<std::complex<ftype>, 1> input_buffer(host_input.data(),
                                                      num_elements);
    if (test_type == placement::OUT_OF_PLACE) {
      output_buffer =
          sycl::buffer<std::complex<ftype>, 1>(buffer.data(), num_elements);
    }

    descriptor<ftype, domain::COMPLEX> desc{
        {static_cast<unsigned long>(params.length)}};
    desc.number_of_transforms = params.batch;
    auto commited_descriptor = desc.commit(queue);

    for (size_t i = 0; i < params.batch; i++)
      reference_forward_dft(host_input, host_reference_output, params.length,
                            i * params.length);
    test_type == placement::OUT_OF_PLACE
        ? commited_descriptor.compute_forward(input_buffer, output_buffer)
        : commited_descriptor.compute_forward(input_buffer);
    queue.wait();
  }
  compare_arrays(test_type == placement::IN_PLACE ? host_input : buffer,
                 host_reference_output, 1e-5);
}

// sizes that use workitem implementation
INSTANTIATE_TEST_SUITE_P(workItemTest, FFTTest,
                         ::testing::ConvertGenerator<param_tuple>(
                             ::testing::Combine(::testing::Values(1, 33, 32000),
                                                ::testing::Range(1, 14))));
// sizes that might use workitem or subgroup implementation depending on device
// and configuration
INSTANTIATE_TEST_SUITE_P(
    workItemOrSubgroupTest, FFTTest,
    ::testing::ConvertGenerator<param_tuple>(
        ::testing::Combine(::testing::Values(1, 3, 555),
                           ::testing::Values(16, 24, 27, 32, 48, 56))));
// sizes that use subgroup implementation
INSTANTIATE_TEST_SUITE_P(
    SubgroupTest, FFTTest,
    ::testing::ConvertGenerator<param_tuple>(::testing::Combine(
        ::testing::Values(1, 3, 555), ::testing::Values(64, 65, 84, 91, 104))));

#endif
