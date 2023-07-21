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

#ifndef PORTFFT_COMMON_UTILS_HPP
#define PORTFFT_COMMON_UTILS_HPP

#include <complex>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

#include <descriptor.hpp>

template <typename T>
void print_vec(std::ostream& stream, const std::vector<T>& vec) {
  stream << "[";
  for (std::size_t i = 0; i < vec.size(); ++i) {
    stream << (i > 0 ? ", " : "") << vec[i];
  }
  stream << "]";
};

/**
 * Return the product of a vector.
 *
 * @tparam T Scalar
 * @param vec vector to multiply
 */
template <typename T>
T prod_vec(const std::vector<T>& vec) {
  return std::accumulate(vec.cbegin(), vec.cend(), T(1), std::multiplies<>());
}

/**
 * Apply strides and distance to a linear packed index to produce a linear unpacked index
 *
 * @param packed_idx Linear packed index of a batched FFT
 * @param fft_length Product of \p lengths to make sure it's not computed in a loop
 * @param lengths Lengths of a multi-dim FFT for a single batch
 * @param strides Strides to apply. The first stride is an offset.
 * @param distance Distance to apply
 */
std::size_t packed_to_unpacked_idx(std::size_t packed_idx, std::size_t fft_length,
                                   const std::vector<std::size_t>& lengths, const std::vector<std::size_t>& strides,
                                   size_t distance) {
  std::size_t unpacked_idx = strides[0];
  std::size_t batch_idx = packed_idx / fft_length;
  unpacked_idx += batch_idx * distance;
  std::size_t fft_idx = packed_idx % fft_length;
  std::size_t dim_len = 1;
  for (std::size_t dim = lengths.size() - 1; dim != std::size_t(-1); --dim) {
    // Input size for the dimension dim
    std::size_t dim_idx = (fft_idx % (dim_len * lengths[dim])) / dim_len;
    unpacked_idx += dim_idx * strides[dim + 1];
    dim_len *= lengths[dim];
  }
  return unpacked_idx;
}

/**
 * Reorder a data container from an input with default stride and distance to an output with custom stride or distance.
 * Called `unpack` as in most cases this leads to a larger output with holes of unused data.
 * Unused data are set to -42 for debugging purposes.
 * With the right stride and distance this can also transpose the data.
 * @see pack_data to reverse the operation.
 *
 * @tparam T Input type
 * @param in Input container
 * @param num_unpack_elements Output length
 * @param lengths FFT lengths
 * @param strides FFT strides starting with an offset
 * @param distance FFT distance i.e. stride between batches
 */
template <typename T>
std::vector<T> unpack_data(const std::vector<T>& in, std::size_t num_unpack_elements,
                           const std::vector<std::size_t>& lengths, const std::vector<std::size_t>& strides,
                           size_t distance) {
  T unused_value = std::is_scalar_v<T> ? T(-42) : T(-42, -42);
  std::vector<T> out(num_unpack_elements, unused_value);
  std::size_t fft_length = prod_vec(lengths);
  for (std::size_t in_idx = 0; in_idx < in.size(); ++in_idx) {
    std::size_t out_idx = packed_to_unpacked_idx(in_idx, fft_length, lengths, strides, distance);
    out[out_idx] = in[in_idx];
  }
  return out;
}

/**
 * Helper function for the main @link unpack_data function
 *
 * @tparam T Input type
 * @tparam Descriptor Descriptor type
 * @param in Input container
 * @param desc Use the given descriptor's lengths, batch, stride and distance
 * @param dir Direction used for the stride and distance
 */
template <typename T, typename Descriptor>
std::vector<T> unpack_data(const std::vector<T>& in, const Descriptor& desc, portfft::direction dir) {
  return unpack_data(in, desc.get_input_count(dir), desc.lengths, desc.get_strides(dir), desc.get_distance(dir));
}

/**
 * Reorder a data container from an input with custom stride or distance to an output with default stride and distance.
 * @see unpack_data to reverse the operation.
 *
 * @tparam T Input type
 * @param in Input container
 * @param batch FFT batch
 * @param lengths FFT lengths
 * @param stride FFT stride starting with an offset
 * @param distance FFT distance i.e. stride between batches
 */
template <typename T>
std::vector<T> pack_data(const std::vector<T>& in, std::size_t batch, const std::vector<std::size_t>& lengths,
                         const std::vector<std::size_t>& stride, size_t distance) {
  std::size_t fft_length = prod_vec(lengths);
  std::size_t num_pack_elements = batch * fft_length;
  std::vector<T> out(num_pack_elements);
  for (std::size_t out_idx = 0; out_idx < num_pack_elements; ++out_idx) {
    std::size_t in_idx = packed_to_unpacked_idx(out_idx, fft_length, lengths, stride, distance);
    out[out_idx] = in[in_idx];
  }
  return out;
}

/**
 * Helper function for the main @link pack_data function
 *
 * @tparam T Input type
 * @tparam Descriptor Descriptor type
 * @param in Input container
 * @param num_unpack_elements Output length
 * @param desc Use the given descriptor's lengths, batch, stride and distance
 * @param dir Direction used for the stride and distance
 */
template <typename T, typename Descriptor>
std::vector<T> pack_data(const std::vector<T>& in, const Descriptor& desc, portfft::direction dir) {
  return pack_data(in, desc.number_of_transforms, desc.lengths, desc.get_strides(dir), desc.get_distance(dir));
}

#endif
