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

#include <sstream>
#include <stdexcept>

namespace portfft {

/**
 * Base exception class for all portFFT exceptions
 */
class base_error : public std::runtime_error {
 private:
  template <typename... Ts>
  std::string concat(const Ts&... args) {
    std::stringstream ss;
    (ss << ... << args);
    return ss.str();
  }

 public:
  template <typename... Ts>
  explicit base_error(const Ts&... args) : std::runtime_error{concat(args...)} {}
};

/**
 * Exception internal to the portFFT library.
 */
struct internal_error : public base_error {
  template <typename... Ts>
  explicit internal_error(const Ts&... args) : base_error(args...) {}
};

/**
 * Exception thrown when the descriptor provided by the user is invalid.
 */
struct invalid_configuration : public base_error {
  template <typename... Ts>
  explicit invalid_configuration(const Ts&... args) : base_error(args...) {}
};

/**
 * Exception thrown if the provided descriptor is not supported at the moment or cannot be supported on a particular
 * device or with the particular CMake configuration.
 */
struct unsupported_configuration : public base_error {
  template <typename... Ts>
  explicit unsupported_configuration(const Ts&... args) : base_error(args...) {}
};

/**
 * Exception class to be thrown when more than available local memory is required
 */
struct out_of_local_memory_error : public unsupported_configuration {
  template <typename... Ts>
  explicit out_of_local_memory_error(const Ts&... args) : unsupported_configuration(args...) {}
};

};  // namespace portfft

#endif
