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

#include <descriptor.hpp>
#include <enums.hpp>

#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>


template <typename T>
void transpose(const T* input, T* output, std::size_t N, std::size_t M) {
  for (std::size_t i = 0; i < N * M; i++) {
    std::size_t j = i / N;
    std::size_t k = i % N;
    output[i] = input[k * M + j];
  }
}

/** Generate input and output reference data to test an FFT against
 * @tparam Dir The direction of the transform
 * @tparam Scalar type of the scalar used for computations
 * @tparam Domain domain of the FFT
 * @param desc The description of the FFT
 * @return a pair of vectors containing potential input and output data for a problem with the given descriptor
 **/
template <portfft::direction Dir, typename Scalar, portfft::domain Domain>
auto gen_fourier_data(portfft::descriptor<Scalar, Domain>& desc) {
  constexpr bool IsRealDomain = Domain == portfft::domain::REAL;

  const auto batches = desc.number_of_transforms;
  const auto& dims = desc.lengths;

  std::string command =
      "python3 -c \""
      "import numpy as np;"
      "from sys import stdout;";
  command.append("batch = ").append(std::to_string(batches)).append(";");
  command.append("dims = [");
  assert(dims.size() > 0);
  command.append(std::to_string(dims[0]));
  for (auto itr = dims.cbegin() + 1; itr < dims.cend(); itr += 1) {
    command.append(",").append(std::to_string(*itr));
  }
  command.append("];");

  command.append(
      "dataGenDims = [batch] + dims;"
      "inData = np.random.uniform(-1, 1, dataGenDims).astype(np.double);");

  if (!IsRealDomain) {
    command.append("inData = inData + 1j * np.random.uniform(-1, 1, dataGenDims).astype(np.double);");
  }

  command.append(
      "outData = np.fft.fftn(inData, axes=range(1, len(dims) + 1));"
      "inData.reshape(-1, 1);"
      "outData.reshape(-1, 1);"
      "stdout.buffer.write(inData.tobytes());"
      "stdout.buffer.write(outData.tobytes());"
      "\"");

  FILE* f = popen(command.c_str(), "r");
  if (f == nullptr) {
    throw std::runtime_error("Command to create reference data failed\n");
  }

  auto elements = std::accumulate(dims.cbegin(), dims.cend(), batches, std::multiplies<>());
  auto backward_elements = elements;
  if (IsRealDomain) {
    backward_elements = std::accumulate(dims.cbegin(), dims.cend() - 1, dims.back() / 2 + 1, std::multiplies<>());
  }

  using FwdDoubleType = typename std::conditional_t<IsRealDomain, double, std::complex<double>>;
  using BwdDoubleType = std::complex<double>;

  std::vector<FwdDoubleType> forward(elements);
  std::vector<BwdDoubleType> backward(backward_elements);

  auto fwd_read = std::fread(forward.data(), sizeof(FwdDoubleType), elements, f);
  if (fwd_read != elements) {
    throw std::runtime_error("Reference data was not transferred correctly");
  }
  auto bwd_read = std::fread(backward.data(), sizeof(BwdDoubleType), backward_elements, f);
  if (bwd_read != backward_elements) {
    throw std::runtime_error("Reference data was not transferred correctly");
  }

  if (pclose(f) != 0) {
    throw std::runtime_error("Failed to close command pipe");
  }

  // cast to the correct type if necessary
  if constexpr (std::is_same_v<Scalar, double>) {
    if constexpr (Dir == portfft::direction::FORWARD) {
      return std::make_pair(forward, backward);
    } else {
      return std::make_pair(backward, forward);
    }
  } else {
    using FwdSingleType = typename std::conditional_t<IsRealDomain, float, std::complex<float>>;
    using BwdSingleType = std::complex<float>;
    std::vector<FwdSingleType> forward_single(elements);
    std::vector<BwdSingleType> backward_single(backward_elements);

    std::copy(forward.cbegin(), forward.cend(), forward_single.begin());
    std::copy(backward.cbegin(), backward.cend(), backward_single.begin());

    if constexpr (Dir == portfft::direction::FORWARD) {
      return std::make_pair(forward_single, backward_single);
    } else {
      return std::make_pair(backward_single, forward_single);
    }
  }
}

/** Test the difference between a dft result and a reference results. Throws an exception if there is a differences.
 * @tparam Dir The direction of the DFT being verified
 * @tparam Scalar type of the scalar used for computations
 * @tparam Domain domain of the FFT
 * @param desc The description of the FFT.
 * @param ref_output_raw The reference data to compare the result with before any transpose is applied
 * @param actual_output The actual result of the computation
 * @param comparison_tolerance An absolute and relative allowed error in the calculation
 **/
template <portfft::direction Dir, typename ElemT, typename Scalar, portfft::domain Domain>
void verify_dft(const portfft::descriptor<Scalar, Domain>& desc, const std::vector<ElemT>& ref_output_raw,
                const std::vector<ElemT>& actual_output, portfft::detail::layout layout_out, const double comparison_tolerance) {
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

  std::size_t dft_len = std::accumulate(data_shape.cbegin(), data_shape.cend(), std::size_t(1), std::multiplies<>());

  // Division by DFT len is required since the reference forward transform has
  // scale factor 1, so inverting with also scale factor 1 would be out by a
  // multiple of dft_len. This scaling is applied to the reference data.
  auto scaling = IsForward ? desc.forward_scale : desc.backward_scale * static_cast<Scalar>(dft_len);

  std::vector<ElemT> ref_output = ref_output_raw;
  if (layout_out == portfft::detail::layout::BATCH_INTERLEAVED) {
    transpose(ref_output_raw.data(), ref_output.data(), desc.number_of_transforms, dft_len);
  } 

  for (std::size_t t = 0; t < desc.number_of_transforms; ++t) {
    const ElemT* this_batch_ref = ref_output.data() + dft_len * t;
    const ElemT* this_batch_computed = actual_output.data() + dft_len * t;

    for (std::size_t e = 0; e != dft_len; ++e) {
      const auto diff = std::abs(this_batch_computed[e] - this_batch_ref[e] * scaling);
      if (diff > comparison_tolerance && diff / std::abs(this_batch_computed[e]) > comparison_tolerance) {
        // std::endl is used intentionally to flush the error message before google test exits the test.
        std::cerr << "transform " << t << ", element " << e << ", with global idx " << t * dft_len + e
                  << ", does not match\nref " << this_batch_ref[e] * scaling << " vs " << this_batch_computed[e]
                  << "\ndiff " << diff << ", tolerance " << comparison_tolerance << std::endl;
        throw std::runtime_error("Verification Failed");
      }
    }
  }
}

#endif  // PORTFFT_COMMON_REFERENCE_DATA_WRANGLER_HPP
