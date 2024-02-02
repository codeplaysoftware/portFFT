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

#ifndef PORTFFT_COMMITTED_DESCRIPTOR_HPP
#define PORTFFT_COMMITTED_DESCRIPTOR_HPP

#include <sycl/sycl.hpp>

#include <complex>
#include <vector>

#include "enums.hpp"

#include "committed_descriptor_impl.hpp"

namespace portfft {

template <typename Scalar, domain Domain>
class committed_descriptor : private detail::committed_descriptor_impl<Scalar, Domain> {
 public:
  /**
   * Alias for `Scalar`.
   */
  using scalar_type = Scalar;

  /**
   * std::complex with `Scalar` scalar.
   */
  using complex_type = std::complex<Scalar>;

  // Use base class constructor
  using detail::committed_descriptor_impl<Scalar, Domain>::committed_descriptor_impl;
  // Use base class function without this->
  using detail::committed_descriptor_impl<Scalar, Domain>::dispatch_direction;

  /**
   * Computes in-place forward FFT, working on a buffer.
   *
   * @param inout buffer containing input and output data
   */
  void compute_forward(sycl::buffer<complex_type, 1>& inout) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    // For now we can just call out-of-place implementation.
    // This might need to be changed once we implement support for large sizes that work in global memory.
    compute_forward(inout, inout);
  }

  /**
   * Computes in-place forward FFT, working on buffers.
   *
   * @param inout_real buffer containing real part of the input and output data
   * @param inout_imag buffer containing imaginary part of the input and output data
   */
  void compute_forward(sycl::buffer<scalar_type, 1>& inout_real, sycl::buffer<scalar_type, 1>& inout_imag) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    // For now we can just call out-of-place implementation.
    // This might need to be changed once we implement support for large sizes that work in global memory.
    compute_forward(inout_real, inout_imag, inout_real, inout_imag);
  }

  /**
   * Computes in-place backward FFT, working on a buffer.
   *
   * @param inout buffer containing input and output data
   */
  void compute_backward(sycl::buffer<complex_type, 1>& inout) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    // For now we can just call out-of-place implementation.
    // This might need to be changed once we implement support for large sizes that work in global memory.
    compute_backward(inout, inout);
  }

  /**
   * Computes in-place backward FFT, working on buffers.
   *
   * @param inout_real buffer containing real part of the input and output data
   * @param inout_imag buffer containing imaginary part of the input and output data
   */
  void compute_backward(sycl::buffer<scalar_type, 1>& inout_real, sycl::buffer<scalar_type, 1>& inout_imag) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    // For now we can just call out-of-place implementation.
    // This might need to be changed once we implement support for large sizes that work in global memory.
    compute_backward(inout_real, inout_imag, inout_real, inout_imag);
  }

  /**
   * Computes out-of-place forward FFT, working on buffers.
   *
   * @param in buffer containing input data
   * @param out buffer containing output data
   */
  void compute_forward(const sycl::buffer<complex_type, 1>& in, sycl::buffer<complex_type, 1>& out) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    dispatch_direction(in, out, in, out, complex_storage::INTERLEAVED_COMPLEX, direction::FORWARD);
  }

  /**
   * Computes out-of-place forward FFT, working on buffers.
   *
   * @param in_real buffer containing real part of the input data
   * @param in_imag buffer containing imaginary part of the input data
   * @param out_real buffer containing real part of the output data
   * @param out_imag buffer containing imaginary part of the output data
   */
  void compute_forward(const sycl::buffer<scalar_type, 1>& in_real, const sycl::buffer<scalar_type, 1>& in_imag,
                       sycl::buffer<scalar_type, 1>& out_real, sycl::buffer<scalar_type, 1>& out_imag) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    dispatch_direction(in_real, out_real, in_imag, out_imag, complex_storage::SPLIT_COMPLEX, direction::FORWARD);
  }

  /**
   * Computes out-of-place forward FFT, working on buffers.
   *
   * @param in buffer containing input data
   * @param out buffer containing output data
   */
  void compute_forward(const sycl::buffer<Scalar, 1>& /*in*/, sycl::buffer<complex_type, 1>& /*out*/) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    throw unsupported_configuration("Real to complex FFTs not yet implemented.");
  }

  /**
   * Compute out of place backward FFT, working on buffers
   *
   * @param in buffer containing input data
   * @param out buffer containing output data
   */
  void compute_backward(const sycl::buffer<complex_type, 1>& in, sycl::buffer<complex_type, 1>& out) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    dispatch_direction(in, out, in, out, complex_storage::INTERLEAVED_COMPLEX, direction::BACKWARD);
  }

  /**
   * Compute out of place backward FFT, working on buffers
   *
   * @param in_real buffer containing real part of the input data
   * @param in_imag buffer containing imaginary part of the input data
   * @param out_real buffer containing real part of the output data
   * @param out_imag buffer containing imaginary part of the output data
   */
  void compute_backward(const sycl::buffer<scalar_type, 1>& in_real, const sycl::buffer<scalar_type, 1>& in_imag,
                        sycl::buffer<scalar_type, 1>& out_real, sycl::buffer<scalar_type, 1>& out_imag) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    dispatch_direction(in_real, out_real, in_imag, out_imag, complex_storage::SPLIT_COMPLEX, direction::BACKWARD);
  }

  /**
   * Computes in-place forward FFT, working on USM memory.
   *
   * @param inout USM pointer to memory containing input and output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_forward(complex_type* inout, const std::vector<sycl::event>& dependencies = {}) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    // For now we can just call out-of-place implementation.
    // This might need to be changed once we implement support for large sizes that work in global memory.
    return compute_forward(inout, inout, dependencies);
  }

  /**
   * Computes in-place forward FFT, working on USM memory.
   *
   * @param inout_real USM pointer to memory containing real part of the input and output data
   * @param inout_imag USM pointer to memory containing imaginary part of the input and output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_forward(scalar_type* inout_real, scalar_type* inout_imag,
                              const std::vector<sycl::event>& dependencies = {}) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    // For now we can just call out-of-place implementation.
    // This might need to be changed once we implement support for large sizes that work in global memory.
    return compute_forward(inout_real, inout_imag, inout_real, inout_imag, dependencies);
  }

  /**
   * Computes in-place forward FFT, working on USM memory.
   *
   * @param inout USM pointer to memory containing input and output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_forward(Scalar* inout, const std::vector<sycl::event>& dependencies = {}) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    // For now we can just call out-of-place implementation.
    // This might need to be changed once we implement support for large sizes that work in global memory.
    return compute_forward(inout, reinterpret_cast<complex_type*>(inout), dependencies);
  }

  /**
   * Computes in-place backward FFT, working on USM memory.
   *
   * @param inout USM pointer to memory containing input and output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_backward(complex_type* inout, const std::vector<sycl::event>& dependencies = {}) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    return compute_backward(inout, inout, dependencies);
  }

  /**
   * Computes in-place backward FFT, working on USM memory.
   *
   * @param inout_real USM pointer to memory containing real part of the input and output data
   * @param inout_imag USM pointer to memory containing imaginary part of the input and output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_backward(scalar_type* inout_real, scalar_type* inout_imag,
                               const std::vector<sycl::event>& dependencies = {}) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    return compute_backward(inout_real, inout_imag, inout_real, inout_imag, dependencies);
  }

  /**
   * Computes out-of-place forward FFT, working on USM memory.
   *
   * @param in USM pointer to memory containing input data
   * @param out USM pointer to memory containing output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_forward(const complex_type* in, complex_type* out,
                              const std::vector<sycl::event>& dependencies = {}) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    return dispatch_direction(in, out, in, out, complex_storage::INTERLEAVED_COMPLEX, direction::FORWARD, dependencies);
  }

  /**
   * Computes out-of-place forward FFT, working on USM memory.
   *
   * @param in_real USM pointer to memory containing real part of the input data
   * @param in_imag USM pointer to memory containing imaginary part of the input data
   * @param out_real USM pointer to memory containing real part of the output data
   * @param out_imag USM pointer to memory containing imaginary part of the output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_forward(const scalar_type* in_real, const scalar_type* in_imag, scalar_type* out_real,
                              scalar_type* out_imag, const std::vector<sycl::event>& dependencies = {}) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    return dispatch_direction(in_real, out_real, in_imag, out_imag, complex_storage::SPLIT_COMPLEX, direction::FORWARD,
                              dependencies);
  }

  /**
   * Computes out-of-place forward FFT, working on USM memory.
   *
   * @param in USM pointer to memory containing input data
   * @param out USM pointer to memory containing output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_forward(const Scalar* /*in*/, complex_type* /*out*/,
                              const std::vector<sycl::event>& /*dependencies*/ = {}) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    throw unsupported_configuration("Real to complex FFTs not yet implemented.");
    return {};
  }

  /**
   * Computes out-of-place backward FFT, working on USM memory.
   *
   * @param in USM pointer to memory containing input data
   * @param out USM pointer to memory containing output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_backward(const complex_type* in, complex_type* out,
                               const std::vector<sycl::event>& dependencies = {}) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    return dispatch_direction(in, out, in, out, complex_storage::INTERLEAVED_COMPLEX, direction::BACKWARD,
                              dependencies);
  }

  /**
   * Computes out-of-place backward FFT, working on USM memory.
   *
   * @param in_real USM pointer to memory containing real part of the input data
   * @param in_imag USM pointer to memory containing imaginary part of the input data
   * @param out_real USM pointer to memory containing real part of the output data
   * @param out_imag USM pointer to memory containing imaginary part of the output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_backward(const scalar_type* in_real, const scalar_type* in_imag, scalar_type* out_real,
                               scalar_type* out_imag, const std::vector<sycl::event>& dependencies = {}) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    return dispatch_direction(in_real, out_real, in_imag, out_imag, complex_storage::SPLIT_COMPLEX, direction::BACKWARD,
                              dependencies);
  }
};

}  // namespace portfft

#endif
