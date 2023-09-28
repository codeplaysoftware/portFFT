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

#ifndef PORTFFT_UNIT_TEST_FFT_TEST_UTILS
#define PORTFFT_UNIT_TEST_FFT_TEST_UTILS

#include <algorithm>
#include <optional>
#include <string>
#include <tuple>

#include <gtest/gtest.h>
#include <portfft.hpp>
#include <sycl/sycl.hpp>

#include "reference_data_wrangler.hpp"
#include "utils.hpp"

// The following file in generated during the build and located at
// ${BUILD_DIR}/ref_data_include/
#include "test_reference.hpp"

using namespace std::complex_literals;
using namespace portfft;

using basic_param_tuple = std::tuple<placement, detail::layout, direction, std::size_t, std::size_t>;
using stride_param_tuple =
    std::tuple<placement, direction, std::size_t, std::size_t, std::vector<std::size_t>, std::vector<std::size_t>>;
using distance_param_tuple = std::tuple<placement, direction, std::size_t, std::size_t, std::size_t, std::size_t>;
using stride_distance_param_tuple = std::tuple<placement, direction, std::size_t, std::size_t, std::vector<std::size_t>,
                                               std::vector<std::size_t>, std::size_t, std::size_t>;

struct test_params {
  placement placement;
  detail::layout layout = detail::layout::UNPACKED;
  direction dir;
  std::size_t batch;
  std::size_t length;
  std::optional<std::vector<std::size_t>> fwd_strides;
  std::optional<std::vector<std::size_t>> bwd_strides;
  std::optional<std::size_t> fwd_distance;
  std::optional<std::size_t> bwd_distance;

  test_params() = default;
  test_params(basic_param_tuple params) : test_params() {
    placement = std::get<0>(params);
    layout = std::get<1>(params);
    dir = std::get<2>(params);
    batch = std::get<3>(params);
    length = std::get<4>(params);
  }
  test_params(stride_param_tuple params) : test_params() {
    placement = std::get<0>(params);
    dir = std::get<1>(params);
    batch = std::get<2>(params);
    length = std::get<3>(params);
    fwd_strides = std::get<4>(params);
    bwd_strides = std::get<5>(params);
    // Set a default distance assuming C2C transforms
    if (batch > 1) {
      descriptor<float, domain::COMPLEX> dummy_desc({length});
      dummy_desc.forward_strides = *fwd_strides;
      dummy_desc.backward_strides = *bwd_strides;
      fwd_distance = dummy_desc.get_input_count(direction::FORWARD);
      bwd_distance = dummy_desc.get_input_count(direction::BACKWARD);
    }
  }
  test_params(distance_param_tuple params) : test_params() {
    placement = std::get<0>(params);
    dir = std::get<1>(params);
    batch = std::get<2>(params);
    length = std::get<3>(params);
    fwd_distance = std::get<4>(params);
    bwd_distance = std::get<5>(params);
  }
  test_params(stride_distance_param_tuple params) : test_params() {
    placement = std::get<0>(params);
    dir = std::get<1>(params);
    batch = std::get<2>(params);
    length = std::get<3>(params);
    fwd_strides = std::get<4>(params);
    bwd_strides = std::get<5>(params);
    fwd_distance = std::get<6>(params);
    bwd_distance = std::get<7>(params);
  }
};

// GTest uses the operator<<< to print this name if the test fails
void operator<<(std::ostream& stream, const test_params& params) {
  stream << "Placement=";
  if (params.placement == placement::IN_PLACE) {
    stream << "IP";
  } else if (params.placement == placement::OUT_OF_PLACE) {
    stream << "OOP";
  }

  stream << ", Layout=";
  if (params.layout == detail::layout::PACKED) {
    stream << "PACKED";
  } else if (params.layout == detail::layout::BATCH_INTERLEAVED) {
    stream << "BATCH_INTERLEAVED";
  } else if (params.layout == detail::layout::UNPACKED) {
    stream << "UNPACKED";
  }

  stream << ", Direction=" << (params.dir == direction::FORWARD ? "Fwd" : "Bwd");
  stream << ", Batch=" << params.batch << ", Length=" << params.length;

  if (params.fwd_strides) {
    stream << ", FwdStrides=";
    print_vec(stream, *params.fwd_strides);
  }
  if (params.bwd_strides) {
    stream << ", BwdStrides=";
    print_vec(stream, *params.bwd_strides);
  }

  if (params.fwd_distance) {
    stream << ", FwdDist=" << *params.fwd_distance;
  }
  if (params.bwd_distance) {
    stream << ", BwdDist=" << *params.bwd_distance;
  }
}

/// Structure used by GTest to generate the test name
struct test_params_print {
  std::string operator()(const testing::TestParamInfo<test_params>& info) const {
    std::stringstream ss;
    ss << info.param;
    std::string test_name = ss.str();
    // Replace and remove invalid characters in GTest names
    std::replace(test_name.begin(), test_name.end(), ',', '_');
    std::replace(test_name.begin(), test_name.end(), ' ', '_');
    std::replace(test_name.begin(), test_name.end(), '=', '_');
    test_name.erase(std::remove(test_name.begin(), test_name.end(), '['), test_name.end());
    test_name.erase(std::remove(test_name.begin(), test_name.end(), ']'), test_name.end());
    return test_name;
  }
};

/**
 * Run some basic checks on the test parameters.
 *
 * @param params Test parameters
 */
void check_test_params(const test_params& params) {
  ASSERT_TRUE(params.length > 0);
  {
    std::vector<std::size_t> instantiated_sizes{PORTFFT_COOLEY_TUKEY_OPTIMIZED_SIZES};
    if (!std::count(instantiated_sizes.cbegin(), instantiated_sizes.cend(), params.length)) {
      GTEST_SKIP();
    }
  }
}

/**
 * Return whether the test should stop after committing the descriptor.
 */
inline bool commit_only_test() {
  std::string test_suite_name = ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();
  // OverlapRead tests are harder to test as the expected result cannot easily be computed from numpy.
  // This is probably an edge-case. We can consider improving these tests if users require this kind of configurations.
  return test_suite_name.find("OverlapRead") != std::string::npos;
}

template <typename FType>
auto get_descriptor(const test_params& params) {
  descriptor<FType, domain::COMPLEX> desc{{params.length}};
  desc.number_of_transforms = params.batch;
  desc.placement = params.placement;
  if (params.layout == detail::layout::PACKED) {
    // Keep default strides and set default distance for the PACKED layout
    if (params.fwd_strides || params.bwd_strides || params.fwd_distance || params.bwd_distance) {
      throw std::runtime_error("Packed layout cannot be used with custom strides or distance");
    }
    desc.forward_distance = desc.get_flattened_length();
    desc.backward_distance = desc.forward_distance;
  } else if (params.layout == detail::layout::BATCH_INTERLEAVED) {
    // Set default strides and distance for the transposed layout
    if (params.fwd_strides || params.bwd_strides || params.fwd_distance || params.bwd_distance) {
      throw std::runtime_error("Transposed layout cannot be used with custom strides or distance");
    }
    desc.forward_strides = {0, static_cast<std::size_t>(params.batch)};
    desc.backward_strides = desc.forward_strides;
    desc.forward_distance = 1;
    desc.backward_distance = 1;
  } else if (params.layout == detail::layout::UNPACKED) {
    // Let the test set any strides or distance
    if (params.fwd_strides) {
      desc.forward_strides = *params.fwd_strides;
    }
    if (params.bwd_strides) {
      desc.backward_strides = *params.bwd_strides;
    }
    if (params.fwd_distance) {
      desc.forward_distance = *params.fwd_distance;
    }
    if (params.bwd_distance) {
      desc.backward_distance = *params.bwd_distance;
    }
  } else {
    throw std::runtime_error("Unsupported layout");
  }
  return desc;
}

/**
 * Common function to initialise USM and buffer tests
 *
 * @tparam FType
 * @param params Test parameters
 *
 * @return Verification spec, descriptor, host input and output buffers, tolerance
 */
template <typename FType>
auto init_test(const test_params& params) {
  auto desc = get_descriptor<FType>(params);
  auto verif_spec = get_matching_spec(verification_data, desc);
  auto host_input = verif_spec.template load_input_data(desc, params.dir);
  std::vector<std::complex<FType>> host_output(desc.get_output_count(params.dir));
  double tolerance = 1e-3;

  return std::make_tuple(verif_spec, desc, host_input, host_output, tolerance);
}

template <typename FType>
void check_fft_usm(const test_params& params, sycl::queue& queue) {
  check_test_params(params);
  auto [verif_spec, desc, host_input, host_output, tolerance] = init_test<FType>(params);
  const bool is_oop = desc.placement == placement::OUT_OF_PLACE;

  std::optional<committed_descriptor<FType, decltype(desc)::Domain>> committed_descriptor_opt;
  try {
    committed_descriptor_opt = std::make_optional(desc.commit(queue));
  } catch (unsupported_configuration& e) {
    GTEST_SKIP() << e.what();
  }
  if (commit_only_test()) {
    return;
  }

  auto committed_descriptor = *committed_descriptor_opt;

  auto device_input = sycl::malloc_device<std::complex<FType>>(host_input.size(), queue);
  std::complex<FType>* device_output = nullptr;
  if (is_oop) {
    device_output = sycl::malloc_device<std::complex<FType>>(host_output.size(), queue);
  }

  auto copy_event = queue.copy(host_input.data(), device_input, host_input.size());

  auto fft_event = [&]() {
    if (is_oop) {
      if (params.dir == direction::FORWARD) {
        return committed_descriptor.compute_forward(device_input, device_output, {copy_event});
      } else {
        return committed_descriptor.compute_backward(device_input, device_output, {copy_event});
      }
    } else {
      if (params.dir == direction::FORWARD) {
        return committed_descriptor.compute_forward(device_input, {copy_event});
      } else {
        return committed_descriptor.compute_backward(device_input, {copy_event});
      }
    }
  }();

  queue.copy(is_oop ? device_output : device_input, host_output.data(), host_output.size(), {fft_event});
  queue.wait();
  verif_spec.verify_dft(committed_descriptor.get_descriptor(), host_output, params.dir, tolerance);

  sycl::free(device_input, queue);
  if (is_oop) {
    sycl::free(device_output, queue);
  }
}

template <typename FType>
void check_fft_buffer(const test_params& params, sycl::queue& queue) {
  check_test_params(params);
  auto [verif_spec, desc, host_input, host_output, tolerance] = init_test<FType>(params);
  const bool is_oop = desc.placement == placement::OUT_OF_PLACE;

  std::optional<committed_descriptor<FType, decltype(desc)::Domain>> committed_descriptor_opt;
  try {
    committed_descriptor_opt = std::make_optional(desc.commit(queue));
  } catch (unsupported_configuration& e) {
    GTEST_SKIP() << e.what();
  }
  if (commit_only_test()) {
    return;
  }

  auto committed_descriptor = *committed_descriptor_opt;

  {
    sycl::buffer<std::complex<FType>, 1> output_buffer(nullptr, 0);
    sycl::buffer<std::complex<FType>, 1> input_buffer(host_input);
    if (is_oop) {
      output_buffer = sycl::buffer<std::complex<FType>, 1>(host_output);
      if (params.dir == direction::FORWARD) {
        committed_descriptor.compute_forward(input_buffer, output_buffer);
      } else {
        committed_descriptor.compute_backward(input_buffer, output_buffer);
      }
    } else {
      if (params.dir == direction::FORWARD) {
        committed_descriptor.compute_forward(input_buffer);
      } else {
        committed_descriptor.compute_backward(input_buffer);
      }
    }
    queue.wait_and_throw();
  }
  verif_spec.verify_dft(committed_descriptor.get_descriptor(), is_oop ? host_output : host_input, params.dir,
                        tolerance);
}

inline void check_invalid_fft(const test_params& params, sycl::queue& queue) {
  auto desc = get_descriptor<float>(params);
  EXPECT_THROW(desc.commit(queue), portfft::invalid_configuration);
}

/**
 * Compare that complex arrays are equal within a tolerance.
 * Only used in tests without a reference data.
 *
 * @tparam T Scalar type
 * @param reference_output Reference output
 * @param device_output Test output
 * @param tol Tolerance
 */
template <typename T>
void compare_arrays(std::vector<std::complex<T>> reference_output, std::vector<std::complex<T>> device_output,
                    double tol) {
  ASSERT_EQ(reference_output.size(), device_output.size());
  for (size_t i = 0; i < reference_output.size(); i++) {
    EXPECT_NEAR(reference_output[i].real(), device_output[i].real(), tol) << "i=" << i;
    EXPECT_NEAR(reference_output[i].imag(), device_output[i].imag(), tol) << "i=" << i;
  }
}

/**
 * Compare that floating arrays are equal within a tolerance.
 * Only used in tests without a reference data.
 *
 * @tparam T Scalar type
 * @param reference_output Reference output
 * @param device_output Test output
 * @param tol Tolerance
 */
template <typename T>
void compare_arrays(std::vector<T> reference_output, std::vector<T> device_output, double tol) {
  ASSERT_EQ(reference_output.size(), device_output.size());
  for (size_t i = 0; i < reference_output.size(); i++) {
    EXPECT_NEAR(reference_output[i], device_output[i], tol) << "i=" << i;
  }
}

#endif
