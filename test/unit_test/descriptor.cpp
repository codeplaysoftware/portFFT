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

#include <gtest/gtest.h>

#include "descriptor.hpp"

#include "utils.hpp"

using Scalar = float;
static constexpr portfft::domain Domain = portfft::domain::COMPLEX;

void test_descriptor_lengths() {
  std::vector<std::size_t> lengths{2, 3};
  portfft::descriptor<Scalar, Domain> desc(lengths);
  EXPECT_EQ(desc.get_flattened_length(), 6);
}

void test_descriptor_strides() {
  std::vector<std::size_t> fwd_strides{2, 2};
  std::vector<std::size_t> bwd_strides{3, 3};
  portfft::descriptor<Scalar, Domain> desc({2, 3});
  desc.forward_strides = fwd_strides;
  desc.backward_strides = bwd_strides;
  EXPECT_EQ(desc.get_strides(portfft::direction::FORWARD), fwd_strides);
  EXPECT_EQ(desc.get_strides(portfft::direction::BACKWARD), bwd_strides);
}

void test_descriptor_distance() {
  std::size_t fwd_distance = 2;
  std::size_t bwd_distance = 3;
  portfft::descriptor<Scalar, Domain> desc({2, 3});
  desc.number_of_transforms = 2;
  desc.forward_distance = fwd_distance;
  desc.backward_distance = bwd_distance;
  EXPECT_EQ(desc.get_distance(portfft::direction::FORWARD), fwd_distance);
  EXPECT_EQ(desc.get_distance(portfft::direction::BACKWARD), bwd_distance);
}

void test_descriptor_scale() {
  Scalar fwd_scale = 2.f;
  Scalar bwd_scale = 3.f;
  portfft::descriptor<Scalar, Domain> desc({2, 3});
  desc.forward_scale = fwd_scale;
  desc.backward_scale = bwd_scale;
  EXPECT_EQ(desc.get_scale(portfft::direction::FORWARD), fwd_scale);
  EXPECT_EQ(desc.get_scale(portfft::direction::BACKWARD), bwd_scale);
}

void test_descriptor_buffer_count() {
  // clang-format off
  // Describe the memory layout for the forward and backward buffers. The notation uses `i` to refer to the linear index of the FFT's element and `b` to the batch index.
  // index:      0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   29
  // fwd buffer: i0b0           i1b0           i2b0      i3b0           i4b0           i5b0 i0b1           i1b1           i2b1      i3b1           i4b1           i5b1
  // bwd buffer: i0b0 i0b1 i3b0 i3b1 i1b0 i1b1 i4b0 i4b1 i2b0 i2b1 i5b0 i5b1
  // clang-format on
  std::size_t batch_size = 2;
  std::vector<std::size_t> lengths{2, 3};
  std::vector<std::size_t> fwd_strides{8, 3};
  std::vector<std::size_t> bwd_strides{2, 4};
  std::size_t fwd_distance = 15;
  std::size_t bwd_distance = 1;
  portfft::descriptor<Scalar, Domain> desc(lengths);
  desc.number_of_transforms = batch_size;
  desc.forward_strides = fwd_strides;
  desc.backward_strides = bwd_strides;
  desc.forward_distance = fwd_distance;
  desc.backward_distance = bwd_distance;

  std::size_t fwd_input_count = desc.get_input_count(portfft::direction::FORWARD);
  std::size_t fwd_output_count = desc.get_output_count(portfft::direction::FORWARD);
  std::size_t bwd_input_count = desc.get_input_count(portfft::direction::BACKWARD);
  std::size_t bwd_output_count = desc.get_output_count(portfft::direction::BACKWARD);
  EXPECT_EQ(fwd_input_count, bwd_output_count);
  EXPECT_EQ(fwd_output_count, bwd_input_count);
  EXPECT_EQ(fwd_input_count, 30);
  EXPECT_EQ(fwd_output_count, 12);
}

TEST(descriptor, lengths) { test_descriptor_lengths(); }
TEST(descriptor, strides) { test_descriptor_strides(); }
TEST(descriptor, distance) { test_descriptor_distance(); }
TEST(descriptor, scale) { test_descriptor_scale(); }
TEST(descriptor, buffer_count) { test_descriptor_buffer_count(); }
