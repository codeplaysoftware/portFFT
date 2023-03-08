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

#ifndef SYCL_FFT_FFT_TEST_UTILS
#define SYCL_FFT_FFT_TEST_UTILS

#include "descriptor.hpp"
#include "utils.hpp"
#include "enums.hpp"

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

using namespace sycl_fft;

class WorkItemTest : public ::testing::TestWithParam<int32_t> {};

// test for out-of-place and in-place ffts.
template <typename ftype, placement test_type>
void check_fft_usm(int32_t length, sycl::queue& queue) {
  ASSERT_TRUE(length > 0);
  std::vector<std::complex<ftype>> host_input(length);
  std::vector<std::complex<ftype>> host_reference_output(length);
  std::vector<std::complex<ftype>> buffer(length);

  auto device_input  = sycl::malloc_device<std::complex<ftype>>(length, queue);
  std::complex<ftype>* device_output = nullptr;
  if(test_type == placement::OUT_OF_PLACE)
    device_output = sycl::malloc_device<std::complex<ftype>>(length, queue);
  populate_with_random<std::complex<ftype>>(host_input, -1, 1);

  auto copy_event = queue.copy(host_input.data(), device_input, length);
  
  descriptor<ftype, domain::COMPLEX> desc{
      {static_cast<unsigned long>(length)}};
  auto commited_descriptor = desc.commit(queue);

  auto fft_event = test_type == placement::OUT_OF_PLACE ? 
                                commited_descriptor.compute_forward(device_input, device_output, {copy_event}) :
                                commited_descriptor.compute_forward(device_input, {copy_event});
  reference_forward_dft(host_input, host_reference_output);
  queue.copy(test_type == placement::OUT_OF_PLACE ? device_output : device_input,
             buffer.data(),
             length, {fft_event});
  queue.wait();
  compare_arrays(host_reference_output, buffer, 1e-5);
}

template <typename ftype, placement test_type>
void check_fft_buffer(int32_t length, sycl::queue& queue) {
  ASSERT_TRUE(length > 0);
  std::vector<std::complex<ftype>> host_input(length);
  std::vector<std::complex<ftype>> host_reference_output(length);
  std::vector<std::complex<ftype>> buffer(length);
  
  populate_with_random<std::complex<ftype>>(host_input, -1, 1);
  {
    sycl::buffer<std::complex<ftype>, 1> output_buffer(nullptr, 0);
    sycl::buffer<std::complex<ftype>, 1> input_buffer(host_input.data(), length);
    if(test_type == placement::OUT_OF_PLACE)
      output_buffer = sycl::buffer<std::complex<ftype>, 1>(buffer.data(), length);

    descriptor<ftype, domain::COMPLEX> desc{{static_cast<unsigned long>(length)}};
    auto commited_descriptor = desc.commit(queue);

    reference_forward_dft(host_input, host_reference_output);
    test_type == placement::OUT_OF_PLACE
        ? commited_descriptor.compute_forward(input_buffer, output_buffer)
        : commited_descriptor.compute_forward(input_buffer);
    queue.wait();
  }
  compare_arrays(test_type == placement::IN_PLACE ? host_input : buffer,
                 host_reference_output, 1e-5);
}

INSTANTIATE_TEST_SUITE_P(workItemTest, WorkItemTest,
                        ::testing::Values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13));
#endif