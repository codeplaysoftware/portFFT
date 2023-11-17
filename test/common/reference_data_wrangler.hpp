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

#ifndef PORTFFT_COMMON_REFERENCE_DATA_WRANGLER_HPP
#define PORTFFT_COMMON_REFERENCE_DATA_WRANGLER_HPP

#include <complex>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <portfft/descriptor.hpp>
#include <portfft/enums.hpp>

/**
 * Runs Out of place transpose
 *
 * @tparam T Input Type
 * @param in input pointer
 * @param dft_len innermost dimension of the input
 * @param batches innermost dimension of the output
 */
template <typename T>
std::vector<T> transpose(const std::vector<T>& in, std::size_t dft_len, std::size_t batches) {
  std::vector<T> out(in.size());
  for (std::size_t j = 0; j < dft_len; j++) {
    for (std::size_t i = 0; i < batches; i++) {
      out.at(i + j * batches) = in.at(j + i * dft_len);
    }
  }
  return out;
}

/** Generate input and output reference data to test an FFT against
 * @tparam Dir The direction of the transform
 * @tparam Scalar type of the scalar used for computations
 * @tparam Domain domain of the FFT
 * @tparam PaddingT type of the padding value
 * @param desc The description of the FFT
 * @param padding_value The value to use in memory locations that are not expected to be read or written.
 * @return a pair of vectors containing potential input and output data for a problem with the given descriptor
 **/
template <portfft::direction Dir, typename Scalar, portfft::domain Domain, typename PaddingT>
auto gen_fourier_data(portfft::descriptor<Scalar, Domain>& desc, portfft::detail::layout layout_in,
                      portfft::detail::layout layout_out, PaddingT padding_value) {
  constexpr bool IsRealDomain = Domain == portfft::domain::REAL;
  constexpr bool IsForward = Dir == portfft::direction::FORWARD;

  const auto batches = desc.number_of_transforms;
  const auto& dims = desc.lengths;

  const char* header =
      "python3 -c \""
      "import numpy as np\n"
      "from sys import stdout\n"
      "def gen_data(batch, dims, is_complex, is_double):\n"
      "  scalar_type = np.double if is_double else np.single\n"
      "  complex_type = np.complex128 if is_double else np.complex64\n"
      "  dataGenDims = [batch] + dims\n"
      "  rng = np.random.Generator(np.random.SFC64(0))\n"
      "  inData = rng.uniform(-1, 1, dataGenDims).astype(scalar_type)\n"
      "  if (is_complex):\n"
      "    inData = inData + 1j * rng.uniform(-1, 1, dataGenDims).astype(scalar_type)\n"
      "  outData = np.fft.fftn(inData, axes=range(1, len(dims) + 1))\n"
      "  inData.reshape(-1, 1)\n"
      "  outData.reshape(-1, 1)\n"
      "  inData = inData.astype(complex_type)\n"
      "  outData = outData.astype(complex_type)\n"
      "  stdout.buffer.write(inData.tobytes())\n"
      "  stdout.buffer.write(outData.tobytes())\n"
      "gen_data(";

  std::stringstream command;
  command << header;

  command << batches << ',';

  assert(dims.size() > 0);
  command << "[" << dims[0];
  for (auto itr = dims.cbegin() + 1; itr < dims.cend(); itr += 1) {
    command << "," << *itr;
  }
  command << "],";

  command << (IsRealDomain ? "False" : "True");

  command << "," << (std::is_same_v<Scalar, float> ? "False" : "True");

  command << ")\"";

  FILE* f = popen(command.str().c_str(), "r");
  if (f == nullptr) {
    throw std::runtime_error("Command to create reference data failed\n");
  }

  auto process_close_func = [](FILE* f) {
    if (pclose(f) != 0) {
      throw std::runtime_error("failed to close validation sub-process");
    }
  };
  std::unique_ptr<FILE, decltype(process_close_func)> file_closer(f, process_close_func);

  // Do not take into account the descriptor's stride, distance or offset to load data from Numpy.
  auto elements = desc.get_flattened_length() * batches;
  auto backward_elements = IsRealDomain ? (elements / dims.back()) * (dims.back() / 2 + 1) : elements;

  using FwdType = typename std::conditional_t<IsRealDomain, Scalar, std::complex<Scalar>>;
  using BwdType = std::complex<Scalar>;

  std::vector<FwdType> forward(elements);
  std::vector<BwdType> backward(backward_elements);

  auto fwd_read = std::fread(forward.data(), sizeof(FwdType), elements, f);
  if (fwd_read != elements) {
    throw std::runtime_error("Reference data was not transferred correctly");
  }
  auto bwd_read = std::fread(backward.data(), sizeof(BwdType), backward_elements, f);
  if (bwd_read != backward_elements) {
    throw std::runtime_error("Reference data was not transferred correctly");
  }

  // modify layout
  if (layout_in == portfft::detail::layout::BATCH_INTERLEAVED) {
    if constexpr (IsForward) {
      forward = transpose(forward, elements / batches, desc.number_of_transforms);
    } else {
      backward = transpose(backward, backward_elements / batches, desc.number_of_transforms);
    }
  }
  if (layout_out == portfft::detail::layout::BATCH_INTERLEAVED) {
    if constexpr (IsForward) {
      backward = transpose(backward, backward_elements / batches, desc.number_of_transforms);
    } else {
      forward = transpose(forward, elements / batches, desc.number_of_transforms);
    }
  }

  // Apply scaling factor to the output
  // Do this before adding offset to avoid scaling the offsets
  if (IsForward) {
    auto scaling_factor = desc.forward_scale;
    std::for_each(backward.begin(), backward.end(), [scaling_factor](auto& x) { x *= scaling_factor; });
  } else {
    // Numpy scales the output by `1/dft_len` for the backward direction and does not support arbitrary scales.
    // We need to multiply by `dft_len` to get an unscaled reference and apply an arbitrary scale to it.
    auto scaling_factor = desc.backward_scale * static_cast<Scalar>(desc.get_flattened_length());
    std::for_each(forward.begin(), forward.end(), [scaling_factor](auto& x) { x *= scaling_factor; });
  }

  auto insert_offset = [=](auto inputVec, std::size_t offset) {
    using InputT = decltype(inputVec);
    std::ptrdiff_t fill_sz = static_cast<std::ptrdiff_t>(offset);
    InputT outputVec(inputVec.size() + offset);
    std::fill(outputVec.begin(), outputVec.begin() + fill_sz, static_cast<typename InputT::value_type>(padding_value));
    std::copy(inputVec.cbegin(), inputVec.cend(), outputVec.begin() + fill_sz);
    return outputVec;
  };

  forward = insert_offset(forward, desc.forward_offset);
  backward = insert_offset(backward, desc.backward_offset);

  // Return a pair in the expected order
  if constexpr (IsForward) {
    return std::make_pair(forward, backward);
  } else {
    return std::make_pair(backward, forward);
  }
}

/** Test the difference between a dft result and a reference results. Throws an exception if there is a differences.
 * @tparam Dir The direction of the DFT being verified
 * @tparam Scalar type of the scalar used for computations
 * @tparam Domain domain of the FFT
 * @param desc The description of the FFT.
 * @param ref_output The reference data to compare the result with before any transpose is applied
 * @param actual_output The actual result of the computation
 * @param comparison_tolerance An absolute and relative allowed error in the calculation
 **/
template <portfft::direction Dir, typename ElemT, typename Scalar, portfft::domain Domain>
void verify_dft(const portfft::descriptor<Scalar, Domain>& desc, std::vector<ElemT> ref_output,
                const std::vector<ElemT>& actual_output, const double comparison_tolerance) {
  constexpr bool IsComplex = Domain == portfft::domain::COMPLEX;
  constexpr bool IsForward = Dir == portfft::direction::FORWARD;
  using BwdType = std::complex<Scalar>;

  // check type of reference is correct
  if constexpr (IsForward || IsComplex) {
    static_assert(std::is_same_v<ElemT, BwdType>, "Expected complex data dft verification.");
  } else {
    static_assert(std::is_same_v<ElemT, Scalar>, "Expected real data type for real backward dft verification.");
  }

  auto data_shape = desc.lengths;

  if constexpr (IsForward && Domain == portfft::domain::REAL) {
    data_shape.back() = data_shape.back() / 2 + 1;
  }

  // TODO: Update this to take into account stride and distance.
  std::size_t dft_len = std::accumulate(data_shape.cbegin(), data_shape.cend(), std::size_t(1), std::multiplies<>());

  auto dft_offset = IsForward ? desc.backward_offset : desc.forward_offset;
  for (std::size_t i = 0; i < dft_offset; ++i) {
    if (ref_output[i] != actual_output[i]) {
      std::cerr << "Incorrectly written value in padding at global idx " << i << ", ref " << ref_output[i] << " vs "
                << actual_output[i] << std::endl;
      throw std::runtime_error("Verification Failed");
    }
  }

  for (std::size_t t = 0; t < desc.number_of_transforms; ++t) {
    const ElemT* this_batch_ref = ref_output.data() + dft_len * t + dft_offset;
    const ElemT* this_batch_computed = actual_output.data() + dft_len * t + dft_offset;

    for (std::size_t e = 0; e != dft_len; ++e) {
      const auto diff = std::abs(this_batch_computed[e] - this_batch_ref[e]);
      if (diff > comparison_tolerance && diff / std::abs(this_batch_computed[e]) > comparison_tolerance) {
        // std::endl is used intentionally to flush the error message before google test exits the test.
        std::cerr << "transform " << t << ", element " << e << ", with global idx " << t * dft_len + e
                  << ", does not match\nref " << this_batch_ref[e] << " vs " << this_batch_computed[e] << "\ndiff "
                  << diff << ", tolerance " << comparison_tolerance << std::endl;
        throw std::runtime_error("Verification Failed");
      }
    }
  }
}

#endif  // PORTFFT_COMMON_REFERENCE_DATA_WRANGLER_HPP
