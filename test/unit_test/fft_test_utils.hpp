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

#include <sycl/sycl.hpp>

#include <optional>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>
#include <portfft/portfft.hpp>

#include "reference_data_wrangler.hpp"
#include "sub_tuple.hpp"

using namespace portfft;

/// Whether to run the test using USM or buffers
enum test_memory { usm, buffer };

/**
 * Pack the placement and input/output layouts together to make it easier with ::testing::Combine
 * Many tests use all combination of placement and {PACKED, BATCH_INTERLEAVED} except for inplace where the input and
 * output layouts must match.
 */
struct test_placement_layouts_params {
  placement placement;
  detail::layout input_layout;
  detail::layout output_layout;
};

using basic_param_tuple = std::tuple<test_placement_layouts_params, direction, std::size_t /*batch_size*/,
                                     std::vector<std::size_t> /*lengths*/>;
using offsets_param_tuple =
    std::tuple<test_placement_layouts_params, direction, std::size_t /*batch_size*/,
               std::vector<std::size_t> /*lengths*/, std::pair<std::size_t, std::size_t> /*offset pair*/>;
using scales_param_tuple =
    std::tuple<test_placement_layouts_params, direction, std::size_t /*batch_size*/,
               std::vector<std::size_t> /*lengths*/, double /*forward_scale*/, double /*backward_scale*/>;
// More tuples can be added here to easily instantiate tests that will require different parameters

struct test_params {
  placement placement;
  detail::layout input_layout;
  detail::layout output_layout;
  direction dir;
  std::size_t batch;
  std::vector<std::size_t> lengths;
  std::optional<double> forward_scale;
  std::optional<double> backward_scale;
  std::optional<std::size_t> forward_offset;
  std::optional<std::size_t> backward_offset;

  test_params() = default;

  explicit test_params(basic_param_tuple params) : test_params() {
    auto placement_layouts = std::get<0>(params);
    placement = placement_layouts.placement;
    input_layout = placement_layouts.input_layout;
    output_layout = placement_layouts.output_layout;
    dir = std::get<1>(params);
    batch = std::get<2>(params);
    lengths = std::get<3>(params);
  }

  explicit test_params(offsets_param_tuple params) : test_params(get_sub_tuple<basic_param_tuple>(params)) {
    forward_offset = std::get<4>(params).first;
    backward_offset = std::get<4>(params).second;
  }

  explicit test_params(scales_param_tuple params) : test_params(get_sub_tuple<basic_param_tuple>(params)) {
    forward_scale = std::get<4>(params);
    backward_scale = std::get<5>(params);
  }
};

/// Structure used by GTest to generate the test name
struct test_params_print {
  std::string operator()(const testing::TestParamInfo<test_params>& info) const {
    auto params = info.param;
    std::stringstream ss;
    auto print_layout = [&ss](detail::layout layout) {
      if (layout == detail::layout::PACKED) {
        ss << "PACKED";
      } else if (layout == detail::layout::BATCH_INTERLEAVED) {
        ss << "BATCH_INTERLEAVED";
      }
    };
    auto print_double = [&](double d) {
      std::string fp_str = std::to_string(d);
      std::replace(fp_str.begin(), fp_str.end(), '-', 'm');
      std::replace(fp_str.begin(), fp_str.end(), '.', '_');
      ss << fp_str;
    };

    ss << "Placement_";
    if (params.placement == placement::IN_PLACE) {
      ss << "IP";
    } else if (params.placement == placement::OUT_OF_PLACE) {
      ss << "OOP";
    }
    ss << "__LayoutIn_";
    print_layout(params.input_layout);
    ss << "__LayoutOut_";
    print_layout(params.output_layout);
    ss << "__Direction_" << (params.dir == direction::FORWARD ? "Fwd" : "Bwd");
    ss << "__Batch_" << params.batch;
    ss << "__Lengths";
    for (std::size_t length : params.lengths) {
      ss << "_" << length;
    }
    if (params.forward_scale) {
      ss << "__FwdScale_";
      print_double(*params.forward_scale);
    }
    if (params.backward_scale) {
      ss << "__BwdScale_";
      print_double(*params.backward_scale);
    }
    if (params.forward_offset) {
      ss << "__FwdOffset_" << *params.forward_offset;
    }
    if (params.backward_offset) {
      ss << "__BwdOffset_" << *params.backward_offset;
    }
    return ss.str();
  }
};

/**
 * Prevent GTest from printing "where GetParam() = 32-byte object [...]"
 * We instead generate a test name that always print all of the param members.
 * This is useful to:
 *   * Always make the test parameters visible even when a test pass and is run without ctest.
 *   * Filter the tests on any parameter.
 */
void operator<<(std::ostream&, const test_params&) {}

/**
 * Create a descriptor from the test parameters
 *
 * @tparam FType Scalar type Float / Double
 * @param params Test parameters
 */
template <typename FType>
auto get_descriptor(const test_params& params) {
  descriptor<FType, domain::COMPLEX> desc{params.lengths};
  desc.number_of_transforms = params.batch;
  desc.placement = params.placement;

  auto apply_layout_for_dir = [&desc, &params](detail::layout layout, direction dir) {
    if (layout == detail::layout::PACKED) {
      // Keep default strides and set default distance for the PACKED layout if needed
      if (desc.number_of_transforms > 1) {
        desc.get_distance(dir) = desc.get_flattened_length();
      }
    } else if (layout == detail::layout::BATCH_INTERLEAVED) {
      // Set default strides and distance for the batch interleaved layout
      desc.get_strides(dir) = {static_cast<std::size_t>(params.batch)};
      desc.get_distance(dir) = 1;
    } else {
      throw std::runtime_error("Unsupported layout");
    }
  };
  // First set input strides and distance if needed then output ones
  apply_layout_for_dir(params.input_layout, params.dir);
  apply_layout_for_dir(params.output_layout, inv(params.dir));

  if (params.forward_scale) {
    desc.forward_scale = static_cast<FType>(*params.forward_scale);
  }
  if (params.backward_scale) {
    desc.backward_scale = static_cast<FType>(*params.backward_scale);
  }
  if (params.forward_offset) {
    desc.forward_offset = *params.forward_offset;
  }
  if (params.backward_offset) {
    desc.backward_offset = *params.backward_offset;
  }
  return desc;
}

/**
 * Runs USM FFT test with the given test parameters
 *
 * @tparam TestMemory Whether to run the test using USM or buffers
 * @tparam Dir FFT direction
 * @tparam DescType Descriptor type
 * @tparam InputFType FFT input type, domain and precision
 * @tparam OutputFType FFT output type, domain and precision
 * @param queue Associated queue
 * @param desc Descriptor matching the test parameters
 * @param host_input FFT input
 * @param host_output Future portFFT output, already allocated of the right size. This is the output for both in-place
 * and out-of-place FFTs.
 * @param host_reference_output Reference output
 * @param tolerance Test tolerance
 */
template <test_memory TestMemory, direction Dir, typename DescType, typename InputFType, typename OutputFType>
std::enable_if_t<TestMemory == test_memory::usm> check_fft(sycl::queue& queue, DescType desc,
                                                           const std::vector<InputFType>& host_input,
                                                           std::vector<OutputFType>& host_output,
                                                           const std::vector<OutputFType>& host_reference_output,
                                                           double tolerance) {
  auto committed_descriptor = desc.commit(queue);

  const bool is_oop = desc.placement == placement::OUT_OF_PLACE;
  auto device_input = sycl::malloc_device<InputFType>(host_input.size(), queue);
  OutputFType* device_output = nullptr;
  sycl::event oop_init_event;
  if (is_oop) {
    device_output = sycl::malloc_device<OutputFType>(host_output.size(), queue);
    oop_init_event = queue.copy(host_output.data(), device_output, host_output.size());
  }

  auto copy_event = queue.copy(host_input.data(), device_input, host_input.size(), {oop_init_event});

  sycl::event fft_event = [&]() {
    if (is_oop) {
      if constexpr (Dir == direction::FORWARD) {
        return committed_descriptor.compute_forward(device_input, device_output, {copy_event});
      } else {
        return committed_descriptor.compute_backward(device_input, device_output, {copy_event});
      }
    } else {
      if constexpr (Dir == direction::FORWARD) {
        return committed_descriptor.compute_forward(device_input, {copy_event});
      } else {
        return committed_descriptor.compute_backward(device_input, {copy_event});
      }
    }
  }();

  queue.copy(is_oop ? device_output : device_input, host_output.data(), host_output.size(), {fft_event});
  queue.wait_and_throw();
  verify_dft<Dir>(desc, host_reference_output, host_output, tolerance);

  sycl::free(device_input, queue);
  if (is_oop) {
    sycl::free(device_output, queue);
  }
}

/**
 * Runs buffer FFT test with the given configuration
 *
 * @tparam TestMemory Whether to run the test using USM or buffers
 * @tparam Dir FFT direction
 * @tparam DescType Descriptor type
 * @tparam InputFType FFT input type, domain and precision
 * @tparam OutputFType FFT output type, domain and precision
 * @param queue Associated queue
 * @param desc Descriptor matching the test configuration
 * @param host_input FFT input. Also used as the output for in-place FFTs.
 * @param host_output Future portFFT output, already allocated of the right size. Unused for in-place FFTs.
 * @param host_reference_output Reference output
 * @param tolerance Test tolerance
 */
template <test_memory TestMemory, direction Dir, typename DescType, typename InputFType, typename OutputFType>
std::enable_if_t<TestMemory == test_memory::buffer> check_fft(sycl::queue& queue, DescType desc,
                                                              std::vector<InputFType>& host_input,
                                                              std::vector<OutputFType>& host_output,
                                                              const std::vector<OutputFType>& host_reference_output,
                                                              double tolerance) {
  auto committed_descriptor = desc.commit(queue);

  const bool is_oop = desc.placement == placement::OUT_OF_PLACE;

  {
    sycl::buffer<InputFType, 1> input_buffer(host_input);
    sycl::buffer<OutputFType, 1> output_buffer(nullptr, 0);
    if (is_oop) {
      // Do not copy back the input to the host
      input_buffer.set_final_data(nullptr);
      output_buffer = sycl::buffer<OutputFType, 1>(host_output);
    }

    if (is_oop) {
      if constexpr (Dir == direction::FORWARD) {
        committed_descriptor.compute_forward(input_buffer, output_buffer);
      } else {
        committed_descriptor.compute_backward(input_buffer, output_buffer);
      }
    } else {
      if constexpr (Dir == direction::FORWARD) {
        committed_descriptor.compute_forward(input_buffer);
      } else {
        committed_descriptor.compute_backward(input_buffer);
      }
    }
  }
  verify_dft<Dir>(desc, host_reference_output, is_oop ? host_output : host_input, tolerance);
}

/**
 * Common function to run tests.
 * Initializes tests and run them for USM or buffer.
 *
 * @tparam TestMemory Whether to run the test using USM or buffers
 * @tparam FType Scalar type Float / Double
 * @tparam Dir FFT direction
 * @param params Test parameters
 */
template <test_memory TestMemory, typename FType, direction Dir>
void run_test(const test_params& params) {
  std::vector<sycl::aspect> queue_aspects;
  if constexpr (std::is_same_v<FType, double>) {
    queue_aspects.push_back(sycl::aspect::fp64);
  }
  if constexpr (TestMemory == test_memory::usm) {
    queue_aspects.push_back(sycl::aspect::usm_device_allocations);
  }
  sycl::queue queue;
  try {
    queue = sycl::queue(sycl::aspect_selector(queue_aspects));
  } catch (sycl::exception& e) {
    GTEST_SKIP() << e.what();
    return;
  }

  auto desc = get_descriptor<FType>(params);

  float padding_value = -5.f;  // Value for memory that isn't written to.
  auto [host_input, host_reference_output] =
      gen_fourier_data<Dir>(desc, params.input_layout, params.output_layout, padding_value);
  decltype(host_reference_output) host_output(desc.get_output_count(params.dir), padding_value);
  double tolerance = 1e-3;

  try {
    check_fft<TestMemory, Dir>(queue, desc, host_input, host_output, host_reference_output, tolerance);
  } catch (out_of_local_memory_error& e) {
    GTEST_SKIP() << e.what();
  }
}

/** Check if arrays are equal, throwing std::runtime_error and printing a message if mismatch.
 * @tparam T Element type to compare
 * @param reference_output Expected results
 * @param device_output The data to test
 */
template <typename T>
void expect_arrays_eq(const std::vector<T>& reference_output, const std::vector<T>& device_output) {
  EXPECT_EQ(reference_output.size(), device_output.size());
  for (std::size_t i = 0; i < reference_output.size(); ++i) {
    if (reference_output[i] != device_output[i]) {
      auto diff = std::abs(reference_output[i] - device_output[i]);
      // std::endl is used intentionally to flush the error message before google test exits the test.
      std::cerr << "element " << i << " does not match\n"
                << "ref " << reference_output[i] << " vs " << device_output[i] << "\n"
                << "diff " << diff << std::endl;
      throw std::runtime_error("Verification Failed");
    }
  }
}

#endif
