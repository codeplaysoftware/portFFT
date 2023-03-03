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

#ifndef WORKITEM_TEST_UTILS
#define WORKITEM_TEST_UTILS

#include "descriptor.hpp"
#include "utils.hpp"

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

using namespace sycl_fft;

class WorkItemTest : public ::testing::TestWithParam<int32_t> {};

template <typename ftype>
void check_fft(int32_t length, sycl::queue& queue) {
  ASSERT_TRUE(length > 0);
  std::vector<std::complex<ftype>> host_input(length);
  std::vector<std::complex<ftype>> host_reference_output(length);
  std::vector<std::complex<ftype>> fbuffer(length);

  populate_with_random(host_input, -0.5, 0.5);

  reference_forward_dft(host_input, host_reference_output);

  descriptor<ftype, domain::COMPLEX> desc_float{
      {static_cast<unsigned long>(length)}};

  auto commited_float = desc_float.commit(queue);

  auto device_finput = sycl::malloc_device<std::complex<ftype>>(length, queue);
  auto device_foutput = sycl::malloc_device<std::complex<ftype>>(length, queue);

  auto copy_event = queue.copy(host_input.data(), device_finput, length);

  auto fft_event = commited_float.compute_forward(device_finput, device_foutput,
                                                  {copy_event});
  queue.copy(device_foutput, fbuffer.data(), length, {fft_event});
  queue.wait();

  compare_arrays(fbuffer, host_reference_output, 1e-4);

  sycl::free(device_finput, queue);
  sycl::free(device_foutput, queue);
}

INSTANTIATE_TEST_CASE_P(workItemTest, WorkItemTest,
                        ::testing::Values(1,2,3,4,5,6,7, 8, 9,10,11,12 13, 16, 24,27,32,48, 56,
                         64, 65, 84, 87, 121, 128, 256, 323, 384, 403, 416));
#endif