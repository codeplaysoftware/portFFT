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

#ifndef PORTFFT_DESCRIPTOR_VALIDATE_HPP
#define PORTFFT_DESCRIPTOR_VALIDATE_HPP

#include <string_view>

#include "common/exceptions.hpp"
#include "common/workitem.hpp"
#include "enums.hpp"
#include "utils.hpp"

namespace portfft::detail::validate {

/**
 * Throw an exception if the lengths are invalid when looked at in isolation.
 *
 * @param lengths the dimensions of the transform
 */
inline void validate_lengths(const std::vector<std::size_t>& lengths) {
  if (lengths.empty()) {
    throw invalid_configuration("Invalid lengths, must have at least 1 dimension");
  }
  for (std::size_t i = 0; i < lengths.size(); ++i) {
    if (lengths[i] == 0) {
      throw invalid_configuration("Invalid lengths[", i, "]=", lengths[i], ", must be positive");
    }
  }
}

/**
 * Throw an exception if the layout is unsupported.
 *
 * @tparam Scalar the scalar type for the transform
 * @param lengths the dimensions of the transform
 * @param forward_layout the layout of the forward domain
 * @param backward_layout the layout of the backward domain
 */
template <typename Scalar>
inline void validate_layout(const std::vector<std::size_t>& lengths, portfft::detail::layout forward_layout,
                            portfft::detail::layout backward_layout) {
  if (lengths.size() > 1) {
    const bool supported_layout =
        forward_layout == portfft::detail::layout::PACKED && backward_layout == portfft::detail::layout::PACKED;
    if (!supported_layout) {
      throw unsupported_configuration("Multi-dimensional transforms are only supported with default data layout");
    }
  }
  if (forward_layout == portfft::detail::layout::UNPACKED || backward_layout == portfft::detail::layout::UNPACKED) {
    if (!portfft::detail::fits_in_wi<Scalar>(lengths.back())) {
      throw unsupported_configuration(
          "Arbitrary strides and distances are only supported for sizes that fit in the registers of a single "
          "work-item");
    }
  }
}

/**
 * Throw an exception if individual stride, distance and number_of_transforms values are invalid/inconsistent.
 *
 * @param lengths the dimensions of the transform
 * @param number_of_transforms the number of batches
 * @param strides the strides between elements in a domain
 * @param distance the distance between batches in a domain
 * @param domain_str a string with the name of the domain being validated
 */
inline void validate_strides_distance_basic(const std::vector<std::size_t>& lengths, std::size_t number_of_transforms,
                                            const std::vector<std::size_t>& strides, std::size_t distance,
                                            const std::string_view domain_str) {
  // Validate stride
  std::size_t expected_num_strides = lengths.size();
  if (strides.size() != expected_num_strides) {
    throw invalid_configuration("Mismatching ", domain_str, " strides length got ", strides.size(), " expected ",
                                expected_num_strides);
  }
  for (std::size_t i = 0; i < strides.size(); ++i) {
    if (strides[i] == 0) {
      throw invalid_configuration("Invalid ", domain_str, " stride[", i, "]=", strides[i], ", must be positive");
    }
  }

  // Validate distance
  if (number_of_transforms > 1 && distance == 0) {
    throw invalid_configuration("Invalid ", domain_str, " distance ", distance, ", must be positive for batched FFTs");
  }
}

/**
 * For multidimensional transforms, check that the strides are large enough so there will not be overlap within a single
 * batch. Throw when the strides are not big enough. This accounts for layouts like batch interleaved.
 *
 * @param lengths the dimensions of the transform
 * @param number_of_transforms the number of batches
 * @param strides the strides between elements in a domain
 * @param distance the distance between batches in a domain
 * @param domain_str a string with the name of the domain being validated
 */
inline void strides_distance_multidim_check(const std::vector<std::size_t>& lengths, std::size_t number_of_transforms,
                                            const std::vector<std::size_t>& strides, std::size_t distance,
                                            const std::string_view domain_str) {
  // Quick check for most common configurations.
  // This check has some false-negative for some impractical configurations.
  // View the output data as a N+1 dimensional tensor for a N-dimension FFT: the number of batch is just another
  // dimension with a stride of 'distance'. This sorts the dimensions from fastest moving (inner-most) to slowest
  // moving (outer-most) and check that the stride of a dimension is large enough to avoid overlapping the previous
  // dimension.
  std::vector<std::size_t> generic_strides = strides;
  std::vector<std::size_t> generic_sizes = lengths;
  if (number_of_transforms > 1) {
    generic_strides.push_back(distance);
    generic_sizes.push_back(number_of_transforms);
  }
  std::vector<std::size_t> indices(generic_sizes.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&](std::size_t a, std::size_t b) { return generic_strides[a] < generic_strides[b]; });

  for (std::size_t i = 1; i < indices.size(); ++i) {
    bool fits_in_next_dim =
        generic_strides[indices[i - 1]] * generic_sizes[indices[i - 1]] <= generic_strides[indices[i]];
    if (!fits_in_next_dim) {
      throw invalid_configuration("Domain ", domain_str,
                                  ": multi-dimension strides are not large enough to avoid overlap");
    }
  }
}

/**
 * Check that batches of 1D FFTs don't overlap.
 *
 * @param lengths the dimensions of the transform
 * @param number_of_transforms the number of batches
 * @param strides the strides between elements in a domain
 * @param distance the distance between batches in a domain
 * @param domain_str a string with the name of the domain being validated
 */
inline void strides_distance_1d_check(const std::vector<std::size_t>& lengths, std::size_t number_of_transforms,
                                      const std::vector<std::size_t>& strides, std::size_t distance,
                                      const std::string_view domain_str) {
  // It helps to think of the 1D transform layed out in 2D with row length of stride, that way each element of a
  // transform will be contiguous down a column.

  // * If there is an index collision between batch N and batch N+M, then there will also be a collision between batch
  // N-1 and batch N+M-1, so if there is any index collision, there will also be one with batch 0 (batch N-N and batch
  // N+M-N).
  // * If an index in a transform mod the stride of the transform is zero, then it would collide with the first batch,
  // given an infinite FFT length. For all elements in a transforms, the index mod stride is the same.
  // * If an element in a batch index collides with another batch, then all previous elements in that batch will also
  // index collide with that batch, so we only need to check the first index of each batch.

  const std::size_t fft_size = lengths[0];
  const std::size_t stride = strides[0];

  const std::size_t first_batch_limit = stride * fft_size;
  if (first_batch_limit <= distance) {
    return;
  }

  for (std::size_t b = 1; b < number_of_transforms;) {
    std::size_t batch_first_idx = b * distance;
    auto column = batch_first_idx % stride;
    if (column == 0) {  // there may be a collision with the first batch
      if (batch_first_idx >= first_batch_limit) {
        // any further batch will only be further way
        return;
      }
      throw invalid_configuration("Domain ", domain_str, ": batch ", b, " collides with first batch at index ",
                                  batch_first_idx);
    }

    // there won't be another collision until the column number is near to stride again, so skip a few
    auto batches_until_new_column = (stride - column) / distance;
    if ((stride - column) % distance != 0) {
      batches_until_new_column += 1;
    }
    b += batches_until_new_column;
  }
}

/**
 * Throw an exception if the given strides and distance are invalid for a single domain.
 *
 * @param lengths the dimensions of the transform
 * @param number_of_transforms the number of batches
 * @param strides the strides between elements in a domain
 * @param distance the distance between batches in a domain
 * @param domain_str a string with the name of the domain being validated
 */
inline void strides_distance_check(const std::vector<std::size_t>& lengths, std::size_t number_of_transforms,
                                   const std::vector<std::size_t>& strides, std::size_t distance,
                                   const std::string_view domain_str) {
  validate_strides_distance_basic(lengths, number_of_transforms, strides, distance, domain_str);
  if (lengths.size() > 1) {
    strides_distance_multidim_check(lengths, number_of_transforms, strides, distance, domain_str);
  } else {
    strides_distance_1d_check(lengths, number_of_transforms, strides, distance, domain_str);
  }
}

/**
 * Throw an exception if the given strides and distances are invalid for either domain.
 *
 * @param place where the result is written with respect to where it is read (in-place vs not in-place)
 * @param lengths the dimensions of the transform
 * @param number_of_transforms the number of batches
 * @param forward_strides the strides between elements in the forward domain
 * @param backward_strides the strides between elements in the backward domain
 * @param forward_distance the distance between batches in the forward domain
 * @param backward_distance the distance between batches in the backward domain
 */
inline void validate_strides_distance(placement place, const std::vector<std::size_t>& lengths,
                                      std::size_t number_of_transforms, const std::vector<std::size_t>& forward_strides,
                                      const std::vector<std::size_t>& backward_strides, std::size_t forward_distance,
                                      std::size_t backward_distance) {
  if (place == placement::IN_PLACE) {
    if (forward_strides != backward_strides) {
      throw invalid_configuration("Invalid forward and backward strides must match for in-place configurations");
    }
    if (forward_distance != backward_distance) {
      throw invalid_configuration("Invalid forward and backward distances must match for in-place configurations");
    }
    strides_distance_check(lengths, number_of_transforms, forward_strides, forward_distance, "forward");
  } else {
    strides_distance_check(lengths, number_of_transforms, forward_strides, forward_distance, "forward");
    strides_distance_check(lengths, number_of_transforms, backward_strides, backward_distance, "backward");
  }
}

/**
 * @brief Check as much as possible if a given descriptor is valid and supported for the current capabilties of portFFT.
 * @details The descriptor can still later be deemed unsupported if it is not immediately obvious. If the descriptor is
 * invalid, it should be reported here or not at all.
 *
 * @param params the final description of the problem.
 * @throws portfft::unsupported_configuration when the configuration is unsupported
 * @throws portfft::invalid_configuration when the configuration is invalid e.g. would cause elements to overlap
 */
template <typename Descriptor>
void validate_descriptor(const Descriptor& params) {
  using namespace portfft;

  if constexpr (Descriptor::Domain == domain::REAL) {
    throw unsupported_configuration("REAL domain is unsupported");
  }

  if (params.number_of_transforms == 0) {
    throw invalid_configuration("Invalid number of transform ", params.number_of_transforms, ", must be positive");
  }

  validate_lengths(params.lengths);
  validate_strides_distance(params.placement, params.lengths, params.number_of_transforms, params.forward_strides,
                            params.backward_strides, params.forward_distance, params.backward_distance);
  validate_layout<typename Descriptor::Scalar>(params.lengths, portfft::detail::get_layout(params, direction::FORWARD),
                                               portfft::detail::get_layout(params, direction::BACKWARD));
}

}  // namespace portfft::detail::validate

#endif
