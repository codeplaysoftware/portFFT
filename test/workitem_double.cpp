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

#include "workitem_test_utils.hpp"

TEST_P(WorkItemTest, USM_C2C_Fwd_Double) {
  int32_t length = GetParam();
  auto queue = get_queue(fp64_selector);
  if (!queue)
    GTEST_SKIP() << "Skipping Test with input type as double. No compatible "
                    "device found\n";

  check_fft<double>(length, queue.value());
}


