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

#include <cerrno>
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

// Used to create padding that is either a scalar or a complex value with equal real and imaginary parts.
template <typename T>
T padding_representation(float p) {
  if constexpr (std::is_floating_point_v<T>) {
    return p;
  } else {
    static_assert(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>);
    return {p, p};
  }
}

/**
 * Reshare the packed reference data to the layout specified in \p desc.
 */
template <typename InType, typename Descriptor>
std::vector<InType> reshape_to_desc(const std::vector<InType>& in, const Descriptor& desc,
                                    portfft::detail::layout layout, portfft::direction dir, float padding_value) {
  const auto flat_len = desc.get_flattened_length();

  // assume we are starting with the packed format of the descriptor
  assert(in.size() == flat_len * desc.number_of_transforms);

  // padding is added during initialization
  std::vector<InType> out(desc.get_input_count(dir), padding_representation<InType>(padding_value));

  const auto offset = static_cast<std::ptrdiff_t>(desc.get_offset(dir));

  if (layout == portfft::detail::layout::PACKED) {
    std::copy(in.cbegin(), in.cend(), out.begin() + offset);
  } else {
    // only handling 1D for now
    assert(desc.lengths.size() == 1);
    const auto stride = desc.get_strides(dir).back();
    const auto distance = desc.get_distance(dir);

    // add strides and distances
    InType const* in_iter = in.data();
    InType* out_batch_iter = out.data() + offset;
    for (std::size_t b = 0; b != desc.number_of_transforms; b += 1) {
      InType* out_transform_iter = out_batch_iter;

      for (std::size_t e = 0; e != flat_len; e += 1) {
        *out_transform_iter = *in_iter;

        in_iter += 1;
        out_transform_iter += stride;
      }

      out_batch_iter += distance;
    }
  }
  return out;
}

/** Generate input and output reference data to test an FFT against
 * @tparam Dir The direction of the transform
 * @tparam Storage complex storage to use
 * @tparam Scalar type of the scalar used for computations
 * @tparam Domain domain of the FFT
 * @param desc The description of the FFT
 * @param padding_value The value to use in memory locations that are not expected to be read or written.
 * @param input_layout layout (PACKED/BATCH_INTERLEAVED) of the input data
 * @param output_layout layout (PACKED/BATCH_INTERLEAVED) of the output data
 * @return a tuple of vectors containing input and output data for a problem with the given descriptor. If `Storage` is
 *interleaved the first two tuple values contain vectors of input and output data and the last two vectors are empty. If
 *`Storage` is split, first two values contain input and output real part and the last two input and output imaginary
 *part of the data.
 **/
template <portfft::direction Dir, portfft::complex_storage Storage, typename Scalar, portfft::domain Domain>
auto gen_fourier_data(portfft::descriptor<Scalar, Domain>& desc, portfft::detail::layout input_layout,
                      portfft::detail::layout output_layout, float padding_value) {
  constexpr bool IsRealDomain = Domain == portfft::domain::REAL;
  constexpr bool IsForward = Dir == portfft::direction::FORWARD;
  constexpr bool IsInterleaved = Storage == portfft::complex_storage::INTERLEAVED_COMPLEX;
  constexpr bool DebugInput = false;

  const auto batches = desc.number_of_transforms;
  const auto& dims = desc.lengths;

  const char* header =
      "python3 -c \""
      "import numpy as np\n"
      "from sys import stdout\n"
      "def gen_data(batch, dims, is_complex, is_double, debug_input):\n"
      "  scalar_type = np.double if is_double else np.single\n"
      "  complex_type = np.complex128 if is_double else np.complex64\n"
      "  forward_type = complex_type if is_complex else scalar_type\n"
      "  dataGenDims = [batch] + dims\n"
      "\n"
      "  if (debug_input):\n"
      "    inData = np.arange(np.prod(dataGenDims)).reshape(dataGenDims).astype(forward_type) + 7j\n"
      "  else:\n"
      "    rng = np.random.Generator(np.random.SFC64(0))\n"
      "    inData = rng.uniform(-1, 1, dataGenDims).astype(scalar_type)\n"
      "    if (is_complex):\n"
      "      inData = inData + 1j * rng.uniform(-1, 1, dataGenDims).astype(scalar_type)\n"
      "\n"
      "  if (is_complex):\n"
      "    outData = np.fft.fftn(inData, axes=range(1, len(dims) + 1))\n"
      "  else:\n"
      "    outData = np.fft.rfftn(inData, axes=range(1, len(dims) + 1))\n"
      "  # outData is always double precision at this point\n"
      "  outData = outData.astype(complex_type)\n"
      "\n"
      "  # input and output shape is irrelevant when outputting the buffer\n"
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

  command << "," << (DebugInput ? "True" : "False");

  command << ")\"";

  FILE* f = popen(command.str().c_str(), "r");
  if (f == nullptr) {
    throw std::runtime_error("Command to create reference data failed\n");
  }

  auto process_close_func = [](FILE* f) {
    if (pclose(f) != 0) {
      // Note: strerror may output "Operation not supported" if the process is closed and we have not read all of its
      // output with std::fread
      throw std::runtime_error("failed to close validation sub-process, errno:" + std::string(std::strerror(errno)));
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

  const auto layout_fwd = IsForward ? input_layout : output_layout;
  const auto layout_bwd = IsForward ? output_layout : input_layout;

  // modify layout
  forward = reshape_to_desc(forward, desc, layout_fwd, portfft::direction::FORWARD, padding_value);
  backward = reshape_to_desc(backward, desc, layout_bwd, portfft::direction::BACKWARD, padding_value);

  std::vector<Scalar> forward_real;
  std::vector<Scalar> backward_real;
  std::vector<Scalar> forward_imag;
  std::vector<Scalar> backward_imag;

  if constexpr (!IsInterleaved) {
    if constexpr (!IsRealDomain) {
      forward_real.reserve(forward.size());
      forward_imag.reserve(forward.size());
      for (auto el : forward) {
        forward_real.push_back(el.real());
        forward_imag.push_back(el.imag());
      }
    } else {
      forward_real = std::move(forward);
    }
    backward_real.reserve(backward.size());
    backward_imag.reserve(backward.size());
    for (auto el : backward) {
      backward_real.push_back(el.real());
      backward_imag.push_back(el.imag());
    }
  }

  // Return a tuple in the expected order
  if constexpr (IsForward) {
    if constexpr (IsInterleaved) {
      return std::make_tuple(std::move(forward), std::move(backward), std::move(forward_imag),
                             std::move(backward_imag));
    } else {
      return std::make_tuple(std::move(forward_real), std::move(backward_real), std::move(forward_imag),
                             std::move(backward_imag));
    }
  } else {
    if constexpr (IsInterleaved) {
      return std::make_tuple(std::move(backward), std::move(forward), std::move(backward_imag),
                             std::move(forward_imag));
    } else {
      return std::make_tuple(std::move(backward_real), std::move(forward_real), std::move(backward_imag),
                             std::move(forward_imag));
    }
  }
}

/** Test the difference between a dft result and a reference results. Throws an exception if there is a differences.
 * @tparam Dir The direction of the DFT being verified
 * @tparam Storage complex storage `ref_output` and `actual_output` are using
 * @tparam Domain domain of the FFT
 * @param desc The description of the FFT.
 * @param ref_output The reference data to compare the result with before any transpose is applied
 * @param actual_output The actual result of the computation
 * @param comparison_tolerance An absolute and relative allowed error in the calculation
 * @param elem_name Name of the component (real/complex) this call is checking. Only used in error messages and only
 * if `Storage` is `SPLIT_COMPLEX`.
 **/
template <portfft::direction Dir, portfft::complex_storage Storage, typename ElemT, typename Scalar,
          portfft::domain Domain>
void verify_dft(const portfft::descriptor<Scalar, Domain>& desc, const std::vector<ElemT>& ref_output,
                const std::vector<ElemT>& actual_output, const double comparison_tolerance,
                const std::vector<ElemT>& ref_output_imag = {}, const std::vector<ElemT>& actual_output_imag = {}) {
  using namespace std::complex_literals;
  constexpr bool IsComplex = Domain == portfft::domain::COMPLEX;
  constexpr bool IsForward = Dir == portfft::direction::FORWARD;
  constexpr bool IsInterleaved = Storage == portfft::complex_storage::INTERLEAVED_COMPLEX;
  using BwdType = std::complex<Scalar>;

  // check type of reference is correct
  if constexpr ((IsForward || IsComplex) && IsInterleaved) {
    static_assert(std::is_same_v<ElemT, BwdType>, "Expected complex data dft verification.");
  } else {
    static_assert(std::is_same_v<ElemT, Scalar>,
                  "Expected real data type for real backward / split complex dft verification.");
  }

  if (ref_output.size() != actual_output.size()) {
    std::cerr << "expect the reference size (" << ref_output.size() << ") and the actual size (" << actual_output.size()
              << ") to be the same." << std::endl;
    throw std::runtime_error("Verification Failed");
  }

  const auto dft_len = desc.get_flattened_length();
  const auto dft_offset = desc.get_offset(inv(Dir));
  const auto dft_stride = desc.get_strides(inv(Dir)).back();
  const auto dft_distance = desc.get_distance(inv(Dir));

  for (std::size_t i = 0; i < dft_offset; ++i) {
    if (ref_output[i] != actual_output[i]) {
      if constexpr (!IsInterleaved) {
        std::cerr << "real part:";
      }
      std::cerr << "Incorrectly written value in padding at global idx " << i << ", ref " << ref_output[i] << " vs "
                << actual_output[i] << std::endl;

      throw std::runtime_error("Verification Failed");
    }
    if constexpr (!IsInterleaved) {
      if (ref_output_imag[i] != actual_output_imag[i]) {
        std::cerr << "imag part:";
        std::cerr << "Incorrectly written value in padding at global idx " << i << ", ref " << ref_output_imag[i]
                  << " vs " << actual_output_imag[i] << std::endl;
      }
    }
  }
  portfft::detail::dump_host("ref_output:", ref_output.data(), ref_output.size());
  portfft::detail::dump_host("actual_output:", actual_output.data(), actual_output.size());
  if constexpr (!IsInterleaved) {
    portfft::detail::dump_host("ref_output_imag:", ref_output_imag.data(), ref_output_imag.size());
    portfft::detail::dump_host("actual_output_imag:", actual_output_imag.data(), actual_output_imag.size());
  }

  Scalar max_L2_rel_err = 0;
  for (std::size_t t = 0; t < desc.number_of_transforms; ++t) {
    const ElemT* this_batch_ref = ref_output.data() + dft_distance * t + dft_offset;
    const ElemT* this_batch_computed = actual_output.data() + dft_distance * t + dft_offset;
    const ElemT* this_batch_ref_imag = ref_output_imag.data() + dft_distance * t + dft_offset;
    const ElemT* this_batch_computed_imag = actual_output_imag.data() + dft_distance * t + dft_offset;

    Scalar L2_err = 0;
    Scalar L2_norm = 0;
    for (std::size_t e = 0; e != dft_len; ++e) {
      const auto batch_offset = e * dft_stride;
      BwdType computed_val = this_batch_computed[batch_offset];
      BwdType ref_val = this_batch_ref[batch_offset];
      if constexpr (!IsInterleaved) {
        computed_val += std::complex<Scalar>(0, this_batch_computed_imag[batch_offset]);
        ref_val += std::complex<Scalar>(0, this_batch_ref_imag[batch_offset]);
      }
      Scalar err = std::abs(computed_val - ref_val);
      Scalar norm_val = std::abs(ref_val);
      L2_err += err * err;
      L2_norm += norm_val * norm_val;
    }
    L2_err = std::sqrt(L2_err);
    L2_norm = std::sqrt(L2_norm);
    Scalar L2_rel_err = L2_err / L2_norm;
    max_L2_rel_err = std::max(max_L2_rel_err, L2_rel_err);
  }
  // set to warning to make it print by default
  PORTFFT_LOG_WARNING("Max (across batches) relative L2 error: ", max_L2_rel_err);

  for (std::size_t i = dft_offset; i < ref_output.size(); i += 1) {
    BwdType ref_val = ref_output[i];
    BwdType computed_val = actual_output[i];
    if constexpr (!IsInterleaved) {
      ref_val += std::complex<Scalar>(0, ref_output_imag[i]);
      computed_val += std::complex<Scalar>(0, actual_output_imag[i]);
    }
    const auto abs_diff = std::abs(computed_val - ref_val);
    const auto rel_diff = abs_diff / std::abs(computed_val);
    if (abs_diff > comparison_tolerance && rel_diff > comparison_tolerance) {
      // std::endl is used intentionally to flush the error message before google test exits the test.
      std::cerr << "value at index " << i << " does not match\nref " << ref_val << " vs " << computed_val << "\ndiff "
                << abs_diff << ", tolerance " << comparison_tolerance << std::endl;
      throw std::runtime_error("Verification Failed");
    }
  }
}

#endif  // PORTFFT_COMMON_REFERENCE_DATA_WRANGLER_HPP
