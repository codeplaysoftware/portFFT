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

#ifndef PORTFFT_COMMON_EXCEPTIONS_HPP
#define PORTFFT_COMMON_EXCEPTIONS_HPP

#include <stdexcept>

namespace portfft {

/**
 * Exception thrown if the provided descriptor is not supported at the moment or cannot be supported on a particular
 * device or with the particular CMake configuration.
 */
struct unsupported_configuration : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

/**
 * Exception class to be thrown when more than available local memory is required
 */
struct inadequate_local_memory_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

};  // namespace portfft

#endif
