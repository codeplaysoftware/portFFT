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

#ifndef SYCLFFT_BENCH_BENCH_UTILS_HPP
#define SYCLFFT_BENCH_BENCH_UTILS_HPP

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "enums.hpp"
#include "reference_dft.hpp"

/**
 * number of runs to do when doing an average of many host runs.
 */
static constexpr std::size_t runs_to_average = 10;

template <typename T_index>
inline T_index get_fwd_per_transform(std::vector<T_index> lengths) {
  return std::accumulate(lengths.begin(), lengths.end(), T_index(1), std::multiplies<T_index>());
}

template <typename forward_type, typename T_index>
inline T_index get_bwd_per_transform(std::vector<T_index> lengths) {
  if constexpr (std::is_same_v<forward_type, float> || std::is_same_v<forward_type, double>) {
    return std::accumulate(lengths.begin(), lengths.end() - 1, lengths.back() / 2 + 1, std::multiplies<T_index>());
  } else {
    static_assert(std::is_same_v<forward_type, std::complex<float>> ||
                  std::is_same_v<forward_type, std::complex<double>>);
    return get_fwd_per_transform<T_index>(lengths);
  }
}

// Handle an exception by passing the message onto `SkipWithError`.
// It is expected that this will be placed so the benchmark ends after this is called,
// allowing the test to exit gracefully with an error message before moving onto the next test.
inline void handle_exception(benchmark::State& state, std::exception& e) {
  std::string msg{"Exception thrown: "};
  msg += e.what();
  state.SkipWithError(msg.c_str());
}

/*
 * Compute the reference DFT and compare it with the provided output
 *
 * @tparam forward_type data type for forward domain
 * @tparam backward_type data type for backward domain
 * @param forward_copy the input that was used
 * @param backward_copy the output that was produced
 * @param length the dimensions of the DFT
 * @param number_of_transforms batch size
 * @param forward_scale scaling applied to the output (backward domain)
 */
template <typename forward_type, typename backward_type>
void verify_dft(forward_type* forward_copy, backward_type* backward_copy, std::vector<std::size_t> lengths,
                std::size_t number_of_transforms, double forward_scale) {
  std::size_t fwd_row_elems = lengths.back();
  std::size_t bwd_row_elems = lengths.back();
  if constexpr (!std::is_same_v<forward_type, backward_type>) {
    bwd_row_elems = bwd_row_elems / 2 + 1;
  }
  std::size_t rows = std::accumulate(lengths.begin(), lengths.end() - 1, 1LU, std::multiplies<std::size_t>());
  std::size_t fwd_per_transform = rows * fwd_row_elems;
  std::size_t bwd_per_transform = rows * bwd_row_elems;

  auto reference_buffer = std::make_unique<backward_type[]>(fwd_per_transform);
  constexpr double comparison_tolerance = 1e-2;
  for (std::size_t t = 0; t < number_of_transforms; ++t) {
    const auto fwd_start = forward_copy + t * fwd_per_transform;

    // generate reference for a single transform
    reference_dft<sycl_fft::direction::FORWARD>(fwd_start, reference_buffer.get(), lengths, forward_scale);

    const auto bwd_start = backward_copy + t * bwd_per_transform;
    // compare
    for (std::size_t r = 0; r != rows; ++r) {
      auto ref_row_start = reference_buffer.get() + r * fwd_row_elems;
      auto actual_row_start = bwd_start + r * bwd_row_elems;
      for (std::size_t e = 0; e != bwd_row_elems; ++e) {
        const auto diff = std::abs(ref_row_start[e] - actual_row_start[e]);
        if (diff > comparison_tolerance) {
          // std::endl is used intentionally to flush the error message before google test exits the test.
          std::cerr << "transform " << t << ", row " << r << ", element " << e << " does not match\nref "
                    << ref_row_start[e] << " vs " << actual_row_start[e] << "\ndiff " << diff << ", tolerance "
                    << comparison_tolerance << std::endl;
          throw std::runtime_error("Verification Failed");
        }
      }
    }
  }
}

#endif  // SYCLFFT_BENCH_BENCH_UTILS_HPP
