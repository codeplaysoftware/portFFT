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

#ifndef PORTFFT_DESCRIPTOR_HPP
#define PORTFFT_DESCRIPTOR_HPP

#include <sycl/sycl.hpp>

#include <complex>
#include <numeric>
#include <vector>

#include "committed_descriptor.hpp"
#include "defines.hpp"
#include "descriptor_validate.hpp"
#include "enums.hpp"

namespace portfft {

/**
 * A descriptor containing FFT problem parameters.
 *
 * @tparam DescScalar type of the scalar used for computations
 * @tparam DescDomain domain of the FFT
 */
template <typename DescScalar, domain DescDomain>
struct descriptor {
  /// Scalar type to determine the FFT precision.
  using Scalar = DescScalar;
  static_assert(std::is_floating_point_v<Scalar>, "Scalar must be a floating point type");

  /**
   * FFT domain.
   * Determines whether the input (resp. output) is real or complex in the forward (resp. backward) direction.
   */
  static constexpr domain Domain = DescDomain;

  /**
   * The lengths in elements of each dimension, ordered from most to least significant (i.e. contiguous dimension last).
   * N-D transforms are supported. Must be specified.
   */
  std::vector<std::size_t> lengths;
  /**
   * A scaling factor applied to the output of forward transforms. Default value is 1.
   */
  Scalar forward_scale = 1;
  /**
   * A scaling factor applied to the output of backward transforms. Default value is 1.
   * NB a forward transform followed by a backward transform with both forward_scale and
   * backward_scale set to 1 will result in the data being scaled by the product of the lengths.
   */
  Scalar backward_scale = 1;
  /**
   * The number of transforms or batches that will be solved with each call to compute_xxxward. Default value
   * is 1.
   */
  std::size_t number_of_transforms = 1;
  /**
   * The data layout of complex values. Default value is complex_storage::INTERLEAVED_COMPLEX.
   * complex_storage::INTERLEAVED_COMPLEX indicates that the real and imaginary part of a complex number is contiguous
   * i.e an Array of Structures. complex_storage::SPLIT_COMPLEX indicates that all the real values are contiguous and
   * all the imaginary values are contiguous i.e. a Structure of Arrays.
   */
  complex_storage complex_storage = complex_storage::INTERLEAVED_COMPLEX;
  /**
   * Indicates if the memory address of the output pointer is the same as the input pointer. Default value is
   * placement::OUT_OF_PLACE. When placement::OUT_OF_PLACE is used, only the out of place compute_xxxward functions can
   * be used and the memory pointed to by the input pointer and the memory pointed to by the output pointer must not
   * overlap at all. When placement::IN_PLACE is used, only the in-place compute_xxxward functions can be used.
   */
  placement placement = placement::OUT_OF_PLACE;
  /**
   * The strides of the data in the forward domain in elements.
   * For offset s0 and distance m, for strides[s1,s2,...,sd] the element in batch b at index [i1,i2,...,id] is located
   * at elems[s0 + m*b + s1*i1 + s2*i2 + ... + sd*id]. The default value for a d-dimensional transform is
   * {prod(lengths[0..d-1]), prod(lengths[0..d-2]), ..., lengths[0]*lengths[1], lengths[0], 1}, where prod is the
   * product. Only the default value is supported for transforms with more than one dimension. Strides do not include
   * the offset.
   */
  std::vector<std::size_t> forward_strides;
  /**
   * The strides of the data in the backward domain in elements.
   * For offset s0 and distance m, for strides[s1,s2,...,sd] the element in batch b at index [i1,i2,...,id] is located
   * at elems[s0 + m*b + s1*i1 + s2*i2 + ... + sd*id]. The default value for a d-dimensional transform is
   * {prod(lengths[0..d-1]), prod(lengths[0..d-2]), ..., lengths[0]*lengths[1], lengths[0], 1}, where prod is the
   * product. Only the default value is supported for transforms with more than one dimension. Strides do not include
   * the offset.
   */
  std::vector<std::size_t> backward_strides;
  /**
   * The number of elements between the first value of each transform in the forward domain. The default value is
   * the product of the lengths. Must be either 1 or the product of the lengths.
   * Only the default value is supported for transforms with more than one dimension.
   * For a d-dimensional transform, exactly one of `forward_strides[d-1]` and `forward_distance` must be 1.
   */
  std::size_t forward_distance = 1;
  /**
   * The number of elements between the first value of each transform in the backward domain. The default value
   * is the product of the lengths. Must be the same as forward_distance.
   * Only the default value is supported for transforms with more than one dimension.
   */
  std::size_t backward_distance = 1;
  /**
   * The number of elements between the start of memory allocation for data in forward domain and the first value
   * to use for FFT computation. The default value is 0.
   */
  std::size_t forward_offset = 0;
  /**
   * The number of elements between the start of memory allocation for data in backward domain and the first value
   * to use for FFT computation. The default value is 0.
   */
  std::size_t backward_offset = 0;
  // TODO: add TRANSPOSE, WORKSPACE and ORDERING if we determine they make sense

  /**
   * Construct a new descriptor object.
   *
   * @param lengths size of the FFT transform
   */
  explicit descriptor(const std::vector<std::size_t>& lengths)
      : lengths(lengths), forward_strides(detail::get_default_strides(lengths)), backward_strides(forward_strides) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    // TODO: properly set default values for distances for real transforms
    std::size_t total_size = get_flattened_length();
    forward_distance = total_size;
    backward_distance = total_size;
  }

  /**
   * Commits the descriptor, precalculating what can be done in advance.
   *
   * @param queue queue to use for computations
   * @return committed_descriptor<Scalar, Domain>
   */
  committed_descriptor<Scalar, Domain> commit(sycl::queue& queue) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    detail::validate::validate_descriptor(*this);
    return {*this, queue};
  }

  /**
   * Get the flattened length of an FFT for a single batch, ignoring strides and distance.
   */
  std::size_t get_flattened_length() const noexcept {
    return std::accumulate(lengths.begin(), lengths.end(), 1LU, std::multiplies<std::size_t>());
  }

  /**
   * Get the size of the input buffer for a given direction in terms of the number of elements.
   * The number of elements is the same irrespective of the FFT domain.
   * Takes into account the lengths, number of transforms, strides and direction.
   *
   * @param dir direction
   */
  std::size_t get_input_count(direction dir) const noexcept {
    return get_buffer_count(get_strides(dir), get_distance(dir), get_offset(dir));
  }

  /**
   * Get the size of the output buffer for a given direction in terms of the number of elements.
   * The number of elements is the same irrespective of the FFT domain.
   * Takes into account the lengths, number of transforms, strides and direction.
   *
   * @param dir direction
   */
  std::size_t get_output_count(direction dir) const noexcept { return get_input_count(inv(dir)); }

  /**
   * Return the strides for a given direction
   *
   * @param dir direction
   */
  const std::vector<std::size_t>& get_strides(direction dir) const noexcept {
    return dir == direction::FORWARD ? forward_strides : backward_strides;
  }

  /**
   * Return a mutable reference to the strides for a given direction
   *
   * @param dir direction
   */
  std::vector<std::size_t>& get_strides(direction dir) noexcept {
    return dir == direction::FORWARD ? forward_strides : backward_strides;
  }

  /**
   * Return the distance for a given direction
   *
   * @param dir direction
   */
  std::size_t get_distance(direction dir) const noexcept {
    return dir == direction::FORWARD ? forward_distance : backward_distance;
  }

  /**
   * Return a mutable reference to the distance for a given direction
   *
   * @param dir direction
   */
  std::size_t& get_distance(direction dir) noexcept {
    return dir == direction::FORWARD ? forward_distance : backward_distance;
  }

  /**
   * Return the offset for a given direction
   *
   * @param dir direction
   */
  std::size_t get_offset(direction dir) const noexcept {
    return dir == direction::FORWARD ? forward_offset : backward_offset;
  }

  /**
   * Return a mutable reference to the offset for a given direction
   *
   * @param dir direction
   */
  std::size_t& get_offset(direction dir) noexcept {
    return dir == direction::FORWARD ? forward_offset : backward_offset;
  }

  /**
   * Return the scale for a given direction
   *
   * @param dir direction
   */
  Scalar get_scale(direction dir) const noexcept { return dir == direction::FORWARD ? forward_scale : backward_scale; }

  /**
   * Return a mutable reference to the scale for a given direction
   *
   * @param dir direction
   */
  Scalar& get_scale(direction dir) noexcept { return dir == direction::FORWARD ? forward_scale : backward_scale; }

 private:
  /**
   * Compute the number of elements required for a buffer with the descriptor's length, number of transforms and the
   * given strides and distance.
   * The number of elements is the same irrespective of the FFT domain.
   *
   * @param strides buffer's strides
   * @param distance buffer's distance
   */
  std::size_t get_buffer_count(const std::vector<std::size_t>& strides, std::size_t distance,
                               std::size_t offset) const noexcept {
    // Compute the last element that can be accessed
    std::size_t last_elt_idx = (number_of_transforms - 1) * distance;
    for (std::size_t i = 0; i < lengths.size(); ++i) {
      last_elt_idx += (lengths[i] - 1) * strides[i];
    }
    return offset + last_elt_idx + 1;
  }
};

}  // namespace portfft

#endif
