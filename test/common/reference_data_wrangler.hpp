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

#include <complex>
#include <exception>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "utils.hpp"

/**
 * Represents a verification data file, with information on the DFT represented, and the path at which to find the file.
 * Also routines for loading the data and using it as a reference.
 */
class verif_data_spec {
 public:
  /**
   * Constructor. Should only be needed by the python scripts that generate the reference data.
   */
  verif_data_spec(const std::vector<std::size_t>& dftSize, std::size_t maxBatch, std::string filePath,
                  portfft::domain domain)
      : dftSize(dftSize), maxBatch(maxBatch), filePath(filePath), domain(domain){};

  // The DFT real size - aka. portfft::descriptor::lengths
  std::vector<std::size_t> dftSize;
  // The number of transforms per compute call.
  std::size_t maxBatch;
  // The path where the reference data is to be found.
  std::string filePath;
  // FFT domain
  portfft::domain domain;

  /** Load input data from the reference file.
   *
   * @tparam The Scalar type of the DFT
   * @tparam The Domain of the DFT
   * @param desc The descriptor that this data will be used for with.
   * @param dir Direction of the DFT.
   * @return Packed input data with batches equal to the descriptor.
   **/
  template <typename Scalar, portfft::domain Domain>
  auto load_input_data(const portfft::descriptor<Scalar, Domain>& desc, portfft::direction dir) {
    using elem_t = std::conditional_t<Domain == portfft::domain::COMPLEX, std::complex<Scalar>, Scalar>;
    check_descriptor(desc);
    const bool isForward = dir == portfft::direction::FORWARD;
    auto rawInputPackedData =
        isForward ? load_time_data(desc.number_of_transforms) : load_fourier_data(desc.number_of_transforms);
    auto packedData = cast_data<elem_t>(rawInputPackedData);
    std::size_t expectedPackedSize = desc.number_of_transforms * prod_vec(isForward ? dftSize : fourier_domain_dims());
    if (packedData.size() != expectedPackedSize) {
      throw std::runtime_error("Unexpected number of input elements");
    }

    if (!portfft::detail::has_default_strides_and_distance(desc, dir)) {
      return unpack_data(packedData, desc, dir);
    }
    return packedData;
  }

  /** Load output data from the reference file (computed from time-domain data with scale=1).
   *
   * @tparam The Scalar type of the DFT
   * @tparam The Domain of the DFT
   * @param desc The descriptor that this data will be used for with.
   * @param dir Direction of the DFT.
   * @return Packed output data with batches equal to the descriptor.
   **/
  template <typename Scalar, portfft::domain Domain>
  std::vector<std::complex<Scalar>> load_output_data(const portfft::descriptor<Scalar, Domain>& desc,
                                                     portfft::direction dir) {
    check_descriptor(desc);
    const bool isForward = dir == portfft::direction::FORWARD;
    auto rawInputData =
        isForward ? load_fourier_data(desc.number_of_transforms) : load_time_data(desc.number_of_transforms);
    auto data = cast_data<std::complex<Scalar>>(rawInputData);
    std::size_t expectedPackedSize = desc.number_of_transforms * prod_vec(isForward ? fourier_domain_dims() : dftSize);
    if (data.size() != expectedPackedSize) {
      throw std::runtime_error("Unexpected number of input elements");
    }
    return data;
  }

  /** The Fourier-domain data shape expected from this data.
   **/
  inline std::vector<std::size_t> fourier_domain_dims() {
    auto res = dftSize;
    if (domain == portfft::domain::REAL) {
      res.back() = (res.back() / 2 + 1) * 2;
    }
    return res;
  }

  /** Verify the output of a DFT that was fed with data from this file.
   * @tparam ElemT The element type input for verification.
   * @tparam Scalar The scalar type of the DFT being checked.
   * @tparam Domain The domain of the DFT being checked.
   * @param desc The descriptor of the DFT being checked.
   * @param hostOutput The data to be checked.
   * @param dir The DFT direction.
   * @param comparisonTolerance The tolerance for error.
   **/
  template <typename ElemT, typename Scalar, portfft::domain Domain>
  void verify_dft(const portfft::descriptor<Scalar, Domain>& desc, const std::vector<ElemT>& hostOutput,
                  portfft::direction dir, const double comparisonTolerance) {
    check_descriptor(desc);

    const bool isForward = dir == portfft::direction::FORWARD;
    bool unpackOutput = !portfft::detail::has_default_strides_and_distance(
        desc, isForward ? portfft::direction::BACKWARD : portfft::direction::FORWARD);
    std::vector<ElemT> packedOutput;
    if (unpackOutput) {
      packedOutput = pack_data(hostOutput, desc, inv(dir));
    }
    auto& actualOutput = unpackOutput ? packedOutput : hostOutput;

    std::size_t descBatches = desc.number_of_transforms;
    auto dataShape = isForward ? dftSize : fourier_domain_dims();
    std::size_t dftLen = prod_vec(dataShape);
    // Division by DFT len is required since the reference forward transform has
    // scale factor 1, so inverting with also scale factor 1 would be out by a
    // multiple of dftLen. This scaling is applied to the reference data.
    auto scaling = isForward ? desc.forward_scale : desc.backward_scale * static_cast<Scalar>(dftLen);
    auto referenceData = load_output_data(desc, dir);
    if (referenceData.size() != actualOutput.size()) {
      std::cerr << "Mismatching reference output size=" << referenceData.size()
                << " and actual output size=" << actualOutput.size() << std::endl;
      throw std::runtime_error("Verification Failed");
    }
    compare_arrays(referenceData.data(), actualOutput.data(), dftLen, descBatches, scaling, comparisonTolerance);
  }

 private:
  // The number of doubles in the input data.
  inline std::size_t input_double_count(std::size_t batch) {
    return batch * prod_vec(dftSize) * (domain == portfft::domain::COMPLEX ? 2 : 1);
  }

  // Cast double data read from file to [float, complex<float>, double, complex<double>]
  template <typename ElemT>
  inline std::vector<ElemT> cast_data(std::vector<double>& in) {
    bool constexpr isSinglePrec = std::is_same_v<ElemT, float> || std::is_same_v<ElemT, std::complex<float>>;
    bool constexpr isComplexT = !(std::is_same_v<ElemT, float> || std::is_same_v<ElemT, double>);
    if (isComplexT && in.size() % 2) {
      throw std::runtime_error("Trying to cast an odd length array to complex!");
    }
    std::size_t outSize = isComplexT ? in.size() / 2 : in.size();
    std::vector<ElemT> outData(outSize);
    using scalar_type = std::conditional_t<isSinglePrec, float, double>;
    for (std::size_t i{0}; i < outSize; ++i) {
      if constexpr (isComplexT) {
        outData[i] = ElemT(static_cast<scalar_type>(in[2 * i]), static_cast<scalar_type>(in[2 * i + 1]));
      } else {
        outData[i] = static_cast<scalar_type>(in[i]);
      }
    }
    return outData;
  }

  /** Load data from the input file.
   * @param start The index of the double to start loading from.
   * @param end The index of the double to end loading from.
   * @return A std::vector<double> of data read for the file.
   */
  inline std::vector<double> load_file_data(std::size_t start, std::size_t end) {
    std::ifstream dataFile(filePath, std::ios_base::in | std::ios_base::binary);
    if (!dataFile.good()) {
      throw std::runtime_error("Could not open reference data file at: " + filePath);
    }
    std::vector<double> data(end - start);
    dataFile.seekg(std::ios_base::beg);
    dataFile.seekg(static_cast<std::streamoff>(start * sizeof(double)));
    dataFile.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>((end - start) * sizeof(double)));
    if (!dataFile.good()) {
      throw std::runtime_error("Failed to read reference data from: " + filePath);
    }
    return data;
  }

  /** Load time-domain data from the input file to a double vector.
   * @param batch The number of batches to read in.
   */
  inline std::vector<double> load_time_data(std::size_t batch) { return load_file_data(0, input_double_count(batch)); }

  /** Load fourier-domain data from the input file to a double vector.
   * @param batch The number of batches to read in.
   */
  inline std::vector<double> load_fourier_data(std::size_t batch) {
    auto inCount = input_double_count(maxBatch);
    auto fourierSizes = fourier_domain_dims();
    std::size_t outputDoubleCount = batch * prod_vec(fourierSizes) * 2;
    return load_file_data(inCount, inCount + outputDoubleCount);
  }

  /** Compare data, throwing std::runtime_error and printing a message if out of spec.
   * @tparam ElemT The element type of the data to compare
   * @tparam ScaleT The type of the reference data scale value.
   * @param referenceData The source of truth
   * @param generatedData The data to test
   * @param dftLen The dimensions of the data per batch
   * @param batchComparisons The number of batches
   * @param scaling What should the reference data be scaled by?
   * @param comparisonTolerance The allowable difference before throwing.
   */
  template <typename ElemT, typename ScaleT>
  void compare_arrays(const ElemT* referenceData, const ElemT* generatedData, std::size_t dftLen,
                      std::size_t batchComparisons, ScaleT scaling, double comparisonTolerance) {
    for (std::size_t t = 0; t < batchComparisons; ++t) {
      const ElemT* thisBatchRef = referenceData + dftLen * t;
      const ElemT* thisBatchComputed = generatedData + dftLen * t;

      for (std::size_t e = 0; e != dftLen; ++e) {
        const auto diff = std::abs(thisBatchComputed[e] - thisBatchRef[e] * scaling);
        if (diff > comparisonTolerance) {
          // std::endl is used intentionally to flush the error message before google test exits the test.
          std::cerr << "transform " << t << ", element " << e << ", with global idx " << t * dftLen
                    << ", does not match\nref " << thisBatchRef[e] * scaling << " vs " << thisBatchComputed[e]
                    << "\ndiff " << diff << ", tolerance " << comparisonTolerance << std::endl;
          throw std::runtime_error("Verification Failed");
        }
      }
    }
  }

  /** Check that the user descriptor matches with the loaded file.
   * Helper function for @link load_input_data and @link load_output_data.
   * @tparam Domain User requested domain
   */
  template <typename Scalar, portfft::domain Domain>
  void check_descriptor(const portfft::descriptor<Scalar, Domain>& desc) {
    if (Domain != domain) {
      auto get_domain_str = [](portfft::domain d) { return d == portfft::domain::COMPLEX ? "COMPLEX" : "REAL"; };
      std::stringstream ss;
      ss << "Mismatching domain, expected " << get_domain_str(domain);
      ss << " but descriptor uses " << get_domain_str(Domain) << " domain.";
      throw std::runtime_error(ss.str());
    }
    if (desc.lengths != dftSize) {
      std::stringstream ss;
      ss << "Mismatching lengths, expected ";
      print_vec(ss, dftSize);
      ss << " but descriptor uses ";
      print_vec(ss, desc.lengths);
      ss << " domain.";
      throw std::runtime_error(ss.str());
    }
    if (desc.number_of_transforms > maxBatch) {
      std::stringstream ss;
      ss << "Too large number of transforms, reference data only supports up to " << maxBatch
         << " batches but descriptor requires " << desc.number_of_transforms;
      throw std::runtime_error(ss.str());
    }
  }
};

/** Find a verif_data_spec that can be used to check an DFT described by a portfft::descriptor.
 * @tparam Scalar The descriptor scalar type.
 * @tparam Domain The descriptor domain.
 * @param verifData The generated verification data array - usually named "verification_data"
 * @param desc The descriptor we want data relevant for.
 */
template <typename Scalar, portfft::domain Domain>
verif_data_spec get_matching_spec(const std::vector<verif_data_spec>& verifData,
                                  portfft::descriptor<Scalar, Domain>& desc) {
  for (auto& spec : verifData) {
    if ((desc.lengths == spec.dftSize) && (desc.number_of_transforms <= spec.maxBatch) && (Domain == spec.domain)) {
      return spec;
    }
  }
  throw std::runtime_error("Couldn't find matching specification.");
}

#endif  // PORTFFT_COMMON_REFERENCE_DATA_WRANGLER_HPP
