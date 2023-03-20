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

#include "fft_test_utils.hpp"

TEST_P(FFTTest, USM_IP_C2C_Fwd_Double) {
  auto param = GetParam();
  auto queue = get_queue(fp64_selector);
  CHECK_QUEUE(queue);
  check_fft_usm<double, placement::IN_PLACE>(param, queue.first.value());
}

TEST_P(FFTTest, USM_OOP_C2C_Fwd_Double) {
  auto param = GetParam();
  auto queue = get_queue(fp64_selector);
  CHECK_QUEUE(queue);
  check_fft_usm<double, placement::OUT_OF_PLACE>(param, queue.first.value());
}

TEST_P(FFTTest, BUFFER_IP_C2C_Fwd_Double) {
  auto param = GetParam();
  auto queue = get_queue(fp64_selector);
  CHECK_QUEUE(queue);
  check_fft_buffer<double, placement::IN_PLACE>(param, queue.first.value());
}

TEST_P(FFTTest, BUFFER_OOP_C2C_Fwd_Double) {
  auto param = GetParam();
  auto queue = get_queue(fp64_selector);
  CHECK_QUEUE(queue);
  check_fft_buffer<double, placement::OUT_OF_PLACE>(param, queue.first.value());
}
