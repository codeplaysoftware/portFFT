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
#include <string>
#include <vector>

/**
 * Represents a verification data file, with information on the DFT represented, and the path at which to find the file.
 * Also routines for loading the data and using it as a reference.
 */
class verif_data_spec {
 public:
  /**
   * Constructor. Should only be needed by the python scripts that generate the reference data.
   */
  verif_data_spec(std::vector<std::size_t> dftSize, std::size_t batch, std::string filePath, portfft::domain domain)
      : dftSize(dftSize), batch(batch), filePath(filePath), domain(domain){};

  // The DFT real size - aka. portfft::descriptor::lengths
  std::vector<std::size_t> dftSize;
  // The number of transforms per compute call.
  std::size_t batch;
  // The path where the reference data is to be found.
  std::string filePath;
  // FFT domain
  portfft::domain domain;

  /** Load time-domain data from the reference file.
   *
   * @tparam The Scalar type of the DFT
   * @tparam The Domain of the DFT
   * @param desc The descriptor that this data will be used for with.
   * @return Linearised time-domain data with batches equal to the descriptor.
   **/
  template <typename Scalar, portfft::domain Domain>
  auto load_data_time(portfft::descriptor<Scalar, Domain>& desc) {
    using elem_t = std::conditional_t<Domain == portfft::domain::COMPLEX, std::complex<Scalar>, Scalar>;
    if (Domain != domain) {
      std::string errorStr = "Tried to read data as incorrect type. ";
      errorStr = errorStr + "Ref data is for " + (domain == portfft::domain::COMPLEX ? "COMPLEX" : "REAL") + " domain.";
      throw std::runtime_error(errorStr);
    }
    auto rawInputData = load_input_data(desc.number_of_transforms);
    auto data = cast_data<elem_t>(rawInputData);
    return data;
  }

  /** Load fourier-domain data from the reference file (computed from time-domain data with scale=1).
   *
   * @tparam The Scalar type of the DFT
   * @tparam The Domain of the DFT
   * @param desc The descriptor that this data will be used for with.
   * @return Linearised fourier-domain data with batches equal to the descriptor.
   **/
  template <typename Scalar, portfft::domain Domain>
  std::vector<std::complex<Scalar>> load_data_fourier(portfft::descriptor<Scalar, Domain>& desc) {
    auto rawInputData = load_output_data(desc.number_of_transforms);
    auto data = cast_data<std::complex<Scalar>>(rawInputData);
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
   * @param hostOutput The data to be checked. Expects that distance between
   * batches == the length of the DFT.
   * @param dir The DFT direction.
   * @param comparisonTolerance The tolerance for error.
   **/
  template <typename ElemT, typename Scalar, portfft::domain Domain>
  void verify_dft(portfft::descriptor<Scalar, Domain>& desc, std::vector<ElemT>& hostOutput, portfft::direction dir,
                  const double comparisonTolerance) {
    if ((desc.lengths != dftSize) || (desc.number_of_transforms > batch) || (Domain != domain)) {
      throw std::runtime_error("Can't use this verification data to verify this DFT!");
    }
    using complex_type = std::complex<Scalar>;
    using forward_type = std::conditional_t<Domain == portfft::domain::COMPLEX, complex_type, Scalar>;
    const bool isForward = dir == portfft::direction::FORWARD;
    std::size_t descBatches = desc.number_of_transforms;
    auto dataShape = isForward ? dftSize : fourier_domain_dims();
    std::size_t dftLen = std::accumulate(dataShape.cbegin(), dataShape.cend(), std::size_t(1), std::multiplies<>());
    // Division by DFT len is required since the reference forward transform has
    // scale factor 1, so inverting with also scale factor 1 would be out by a
    // multiple of dftLen. This scaling is applied to the reference data.
    auto scaling = isForward ? desc.forward_scale : desc.backward_scale * static_cast<Scalar>(dftLen);
    if (isForward) {
      auto referenceData = load_data_fourier(desc);
      if constexpr (std::is_same_v<complex_type, ElemT>) {
        compare_arrays(referenceData.data(), hostOutput.data(), dftLen, descBatches, scaling, comparisonTolerance);
      } else {
        throw std::runtime_error("Expected real input data type for forward dft verification.");
      }
    } else {
      auto referenceData = load_data_time(desc);
      if constexpr (std::is_same_v<forward_type, ElemT>) {
        compare_arrays(referenceData.data(), hostOutput.data(), dftLen, descBatches, scaling, comparisonTolerance);
      } else {
        throw std::runtime_error("Expected complex input data type for backward dft verification.");
      }
    }
  }

 private:
  // The number of doubles in the input data.
  inline std::size_t input_double_count() {
    return batch * std::accumulate(dftSize.cbegin(), dftSize.cend(), std::size_t(1), std::multiplies<>()) *
           (domain == portfft::domain::COMPLEX ? 2 : 1);
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
   * @param batchCount The number of batches to read in.
   */
  inline std::vector<double> load_input_data(std::size_t batchCount) {
    if (batchCount > batch) {
      throw std::runtime_error("Requested more batches than promised by specification.");
    }
    return load_file_data(
        0, batchCount * std::accumulate(dftSize.cbegin(), dftSize.cend(), std::size_t(1), std::multiplies<>()) *
               (domain == portfft::domain::COMPLEX ? 2 : 1));
  }

  /** Load fourier-domain data from the input file to a double vector.
   * @param batchCount The number of batches to read in.
   */
  inline std::vector<double> load_output_data(std::size_t batchCount) {
    if (batchCount > batch) {
      throw std::runtime_error("Requested more batches than promised by specification.");
    }
    auto inCount = input_double_count();
    auto fourierSize = fourier_domain_dims();
    std::size_t outputDoubleCount =
        batchCount * std::accumulate(fourierSize.cbegin(), fourierSize.cend(), std::size_t(1), std::multiplies<>()) * 2;
    return load_file_data(inCount, inCount + outputDoubleCount);
  }

  /** Compare data, throwing std::runime_error and printing a message if out of spec.
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
  void compare_arrays(ElemT* referenceData, ElemT* generatedData, std::size_t dftLen, std::size_t batchComparisons,
                      ScaleT scaling, double comparisonTolerance) {
    for (std::size_t t = 0; t < batchComparisons; ++t) {
      const ElemT* thisBatchRef = referenceData + dftLen * t;
      const ElemT* thisBatchComputed = generatedData + dftLen * t;

      for (std::size_t e = 0; e != dftLen; ++e) {
        const auto diff = std::abs(thisBatchComputed[e] - thisBatchRef[e] * scaling);
        if (diff > comparisonTolerance && diff / std::abs(thisBatchComputed[e]) > comparisonTolerance) {
          // std::endl is used intentionally to flush the error message before google test exits the test.
          std::cerr << "transform " << t << ", element " << e << ", with global idx " << t * dftLen
                    << ", does not match\nref " << thisBatchRef[e] * scaling << " vs " << thisBatchComputed[e]
                    << "\ndiff " << diff << ", tolerance " << comparisonTolerance << std::endl;
          throw std::runtime_error("Verification Failed");
        }
      }
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
    if ((desc.lengths == spec.dftSize) && (desc.number_of_transforms <= spec.batch) && (Domain == spec.domain)) {
      return spec;
    }
  }
  throw std::runtime_error("Couldn't find matching specification.");
}

#endif  // PORTFFT_COMMON_REFERENCE_DATA_WRANGLER_HPP
