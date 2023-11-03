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

#ifndef PORTFFT_ENUMS_HPP
#define PORTFFT_ENUMS_HPP

namespace portfft {

enum class domain { REAL, COMPLEX };

enum class complex_storage { COMPLEX, REAL_REAL };

enum class placement { IN_PLACE, OUT_OF_PLACE };

enum class direction { FORWARD, BACKWARD };

/**
 * Return the opposite direction.
 * Useful to get the output of descriptor::get_strides, descriptor::get_distance, or similar functions.
 * @param dir Direction
 */
constexpr direction inv(direction dir) { return dir == direction::FORWARD ? direction::BACKWARD : direction::FORWARD; }

namespace detail {
enum class pad { DO_PAD, DONT_PAD };

enum class level { WORKITEM, SUBGROUP, WORKGROUP, DEVICE };

enum class layout {
  /// Packed layout represents default strides and distance.
  /// Each FFT is contiguous and each FFT is stored one after the other.
  /// dftInput[Idx, BatchId] = ptr[Idx + InputSize * BatchId]
  PACKED,
  /// Unpacked layout represents arbitrary strides or distance.
  // TODO: Add UNPACKED once stride and distance are supported
  // UNPACKED,
  /// Batch interleaved is a special case of unpacked with distance=1 stride=[0, batch_size] which can be better
  /// optimized than the general case.
  /// dftInput[Idx, BatchId] = ptr[Idx * BatchCount + BatchId]
  BATCH_INTERLEAVED
};

enum class memory { BUFFER, USM };

enum class transfer_direction {
  LOCAL_TO_PRIVATE,
  PRIVATE_TO_LOCAL,
  PRIVATE_TO_GLOBAL,
  LOCAL_TO_GLOBAL,
  GLOBAL_TO_LOCAL
};

enum class elementwise_multiply { APPLIED, NOT_APPLIED };

enum class apply_scale_factor { APPLIED, NOT_APPLIED };
}  // namespace detail

}  // namespace portfft

#endif
