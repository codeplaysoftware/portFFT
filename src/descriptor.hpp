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

#ifndef PORTFFT_DESCRIPTOR_HPP
#define PORTFFT_DESCRIPTOR_HPP

#include <common/cooley_tukey_compiled_sizes.hpp>
#include <common/exceptions.hpp>
#include <common/subgroup.hpp>
#include <enums.hpp>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <type_traits>
#include <vector>

namespace portfft {

namespace detail {

// kernel names
// TODO: Remove LayoutIn once we use can use spec constant instead
template <typename Scalar, domain Domain, direction Dir, detail::memory, detail::layout LayoutIn, int SubgroupSize>
class workitem_kernel;
template <typename Scalar, domain Domain, direction Dir, detail::memory, detail::layout LayoutIn, int SubgroupSize>
class subgroup_kernel;
template <typename Scalar, domain Domain, direction Dir, detail::memory, detail::layout LayoutIn, int SubgroupSize>
class workgroup_kernel;

/**
 * Return whether the given descriptor has default strides and distance for a given direction
 *
 * @tparam Descriptor Descriptor type
 * @param desc Descriptor to check
 * @param dir Direction
 */
template <typename Descriptor>
bool has_default_strides_and_distance(const Descriptor& desc, direction dir) {
  // Creating a descriptor will set default strides and distance
  Descriptor defaultDesc(desc.lengths, desc.number_of_transforms);
  return desc.get_strides(dir) == defaultDesc.get_strides(dir) &&
         desc.get_distance(dir) == defaultDesc.get_distance(dir);
}

template <typename Descriptor>
layout get_layout(const Descriptor& desc, direction dir) {
  if (has_default_strides_and_distance(desc, dir)) {
    return detail::layout::PACKED;
  }
  auto batch = desc.number_of_transforms;
  const auto& strides = desc.get_strides(dir);
  auto distance = desc.get_distance(dir);
  // For now require the offset to be 0 to consider this direction to be transposed.
  // This could be relaxed if needed.
  if (batch > 1 && distance == 1 && strides.size() == 2 && strides[0] == 0 && strides[1] == batch) {
    return detail::layout::BATCH_INTERLEAVED;
  }
  return detail::layout::UNPACKED;
}

}  // namespace detail

// forward declaration
template <typename Scalar, domain Domain>
struct descriptor;

/*
Compute functions in the `committed_descriptor` call `dispatch_kernel` and `dispatch_kernel_helper`. These two functions
ensure the kernel is run with a supported subgroup size. Next `dispatch_kernel_helper` calls `run_kernel`. The
`run_kernel` member function picks appropriate implementation and calls the static `run_kernel of that implementation`.
The implementation specific `run_kernel` handles differences between forward and backward computations, casts the memory
(USM or buffers) from complex to scalars and launches the kernel. Each function described in this doc has only one
templated overload that handles both directions of transforms and buffer and USM memory.

Device functions make no assumptions on the size of a work group or the number of workgroups in a kernel. These numbers
can be tuned for each device.

Implementation-specific `run_kernel` function make the size of the FFT that is handled by the individual workitems
compile time constant. The one for subgroup implementation also calls `cross_sg_dispatcher` that makes the
cross-subgroup factor of FFT size compile time constant. They do that by using a switch on the FFT size for one
workitem, before calling `workitem_impl`, `subgroup_impl` or `workgroup_impl` . The `_impl` functions take the FFT size
for one workitem as a template  parameter. Only the calls that are determined to fit into available registers (depending
on the value of PORTFFT_TARGET_REGS_PER_WI macro) are actually instantiated.

The `_impl` functions iterate over the batch of problems, loading data for each first in
local memory then from there into private one. This is done in these two steps to avoid non-coalesced global memory
accesses. `workitem_impl` loads one problem per workitem, `subgroup_impl` loads one problem per subgroup and
`workgroup_impl` loads one problem per workgroup. After doing computations by the calls to `wi_dft` for workitem,
`sg_dft` for subgroup and `wg_dft` for workgroup, the data is written out, going through local memory again.

The computational parts of the implementations are further documented in files with their implementations
`workitem.hpp`, `subgroup.hpp` and `workgroup.hpp`.
*/

/**
 * A committed descriptor that contains everything that is needed to run FFT.
 *
 * @tparam Scalar type of the scalar used for computations
 * @tparam Domain domain of the FFT
 */
template <typename Scalar, domain Domain>
class committed_descriptor {
  using complex_type = std::complex<Scalar>;

  friend struct descriptor<Scalar, Domain>;

  descriptor<Scalar, Domain> params;
  sycl::queue queue;
  sycl::device dev;
  sycl::context ctx;
  std::size_t n_compute_units = 0;
  std::vector<std::size_t> supported_sg_sizes;
  int used_sg_size = 0;
  std::shared_ptr<Scalar> twiddles_forward;
  detail::level level;
  std::vector<int> factors;
  std::optional<sycl::kernel_bundle<sycl::bundle_state::executable>> exec_bundle;
  std::size_t num_sgs_per_wg = 0;
  std::size_t local_memory_size = 0;
  std::string fwd_validation_err;
  std::string bwd_validation_err;

  template <typename Impl, typename... Args>
  auto dispatch(Args&&... args) {
    switch (level) {
      case detail::level::WORKITEM:
        return Impl::template inner<detail::level::WORKITEM, void>::execute(*this, args...);
      case detail::level::SUBGROUP:
        return Impl::template inner<detail::level::SUBGROUP, void>::execute(*this, args...);
      case detail::level::WORKGROUP:
        return Impl::template inner<detail::level::WORKGROUP, void>::execute(*this, args...);
      default:
        throw internal_error("Unimplemented!");
    }
  }

  template <typename Impl, detail::layout LayoutIn, typename... Args>
  auto dispatch(Args&&... args) {
    switch (level) {
      case detail::level::WORKITEM:
        return Impl::template inner<detail::level::WORKITEM, LayoutIn, void>::execute(*this, args...);
      case detail::level::SUBGROUP:
        return Impl::template inner<detail::level::SUBGROUP, LayoutIn, void>::execute(*this, args...);
      case detail::level::WORKGROUP:
        return Impl::template inner<detail::level::WORKGROUP, LayoutIn, void>::execute(*this, args...);
      default:
        throw internal_error("Unimplemented!");
    }
  }

  /**
   * Get kernel ids for the implementation used.
   *
   * @tparam kernel which base template for kernel to use
   * @tparam SubgroupSize size of the subgroup
   * @param ids vector of kernel ids
   */
  template <template <typename, domain, direction, detail::memory, detail::layout, int> class Kernel, int SubgroupSize>
  void get_ids(std::vector<sycl::kernel_id>& ids) {
// if not used, some kernels might be optimized away in AOT compilation and not available here
#define PORTFFT_GET_ID(DIRECTION, MEMORY, TRANSPOSE)                                                          \
  try {                                                                                                       \
    ids.push_back(sycl::get_kernel_id<Kernel<Scalar, Domain, DIRECTION, MEMORY, TRANSPOSE, SubgroupSize>>()); \
  } catch (...) {                                                                                             \
  }

    PORTFFT_GET_ID(direction::FORWARD, detail::memory::BUFFER, detail::layout::PACKED)
    PORTFFT_GET_ID(direction::BACKWARD, detail::memory::BUFFER, detail::layout::PACKED)
    PORTFFT_GET_ID(direction::FORWARD, detail::memory::USM, detail::layout::PACKED)
    PORTFFT_GET_ID(direction::BACKWARD, detail::memory::USM, detail::layout::PACKED)
    PORTFFT_GET_ID(direction::FORWARD, detail::memory::BUFFER, detail::layout::UNPACKED)
    PORTFFT_GET_ID(direction::BACKWARD, detail::memory::BUFFER, detail::layout::UNPACKED)
    PORTFFT_GET_ID(direction::FORWARD, detail::memory::USM, detail::layout::UNPACKED)
    PORTFFT_GET_ID(direction::BACKWARD, detail::memory::USM, detail::layout::UNPACKED)
    PORTFFT_GET_ID(direction::FORWARD, detail::memory::BUFFER, detail::layout::BATCH_INTERLEAVED)
    PORTFFT_GET_ID(direction::BACKWARD, detail::memory::BUFFER, detail::layout::BATCH_INTERLEAVED)
    PORTFFT_GET_ID(direction::FORWARD, detail::memory::USM, detail::layout::BATCH_INTERLEAVED)
    PORTFFT_GET_ID(direction::BACKWARD, detail::memory::USM, detail::layout::BATCH_INTERLEAVED)

#undef PORTFFT_GET_ID
  }

  /**
   * Prepares the implementation for the particular problem size. That includes factorizing it and getting ids for the
   * set of kernels that need to be JIT compiled.
   *
   * @tparam SubgroupSize size of the subgroup
   * @param[out] ids list of kernel ids that need to be JIT compiled
   * @return detail::level
   */
  template <int SubgroupSize>
  detail::level prepare_implementation(std::vector<sycl::kernel_id>& ids) {
    factors.clear();
    std::size_t fft_size = params.lengths[0];
    if (!detail::cooley_tukey_size_list_t::has_size(fft_size)) {
      throw unsupported_configuration("FFT size ", fft_size, " is not supported!");
    }
    if (detail::fits_in_wi<Scalar>(fft_size)) {
      get_ids<detail::workitem_kernel, SubgroupSize>(ids);
      return detail::level::WORKITEM;
    }
    int factor_sg = detail::factorize_sg(static_cast<int>(fft_size), SubgroupSize);
    int factor_wi = static_cast<int>(fft_size) / factor_sg;
    if (detail::fits_in_sg<Scalar>(fft_size, SubgroupSize)) {
      // This factorization is duplicated in the dispatch logic on the device.
      // The CT and spec constant factors should match.
      factors.push_back(factor_wi);
      factors.push_back(factor_sg);
      get_ids<detail::subgroup_kernel, SubgroupSize>(ids);
      return detail::level::SUBGROUP;
    }
    std::size_t N = detail::factorize(fft_size);
    std::size_t M = fft_size / N;
    int factor_sg_N = detail::factorize_sg(static_cast<int>(N), SubgroupSize);
    int factor_wi_N = static_cast<int>(N) / factor_sg_N;
    int factor_sg_M = detail::factorize_sg(static_cast<int>(M), SubgroupSize);
    int factor_wi_M = static_cast<int>(M) / factor_sg_M;
    if (detail::fits_in_wi<Scalar>(factor_wi_N) && detail::fits_in_wi<Scalar>(factor_wi_M)) {
      factors.push_back(factor_wi_N);
      factors.push_back(factor_sg_N);
      factors.push_back(factor_wi_M);
      factors.push_back(factor_sg_M);
      // This factorization of N and M is duplicated in the dispatch logic on the device.
      // The CT and spec constant factors should match.
      get_ids<detail::workgroup_kernel, SubgroupSize>(ids);
      return detail::level::WORKGROUP;
    }
    // TODO global
    throw unsupported_configuration("FFT size ", fft_size, " is not supported!");
  }

  /**
   * Struct for dispatching `set_spec_constants()` call.
   */
  struct set_spec_constants_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template <detail::level lev, typename Dummy>
    struct inner {
      static void execute(committed_descriptor& desc, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle);
    };
  };

  /**
   * Sets the implementation dependant specialization constant values.
   *
   * @param in_bundle kernel bundle to set the specialization constants on
   */
  void set_spec_constants(sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle) {
    dispatch<set_spec_constants_struct>(in_bundle);
  }

  /**
   * Struct for dispatching `num_scalars_in_local_mem()` call.
   */
  struct num_scalars_in_local_mem_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template <detail::level lev, typename Dummy>
    struct inner {
      static std::size_t execute(committed_descriptor& desc, direction dir);
    };
  };

  /**
   * Determine the number of scalars we need to have space for in the local memory. It may also modify `num_sgs_in_wg`
   * to make the problem fit in the local memory.
   *
   * @param dir Direction
   * @return std::size_t the number of scalars
   */
  std::size_t num_scalars_in_local_mem(direction dir) { return dispatch<num_scalars_in_local_mem_struct>(dir); }

  /**
   * Struct for dispatching `calculate_twiddles()` call.
   */
  struct calculate_twiddles_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template <detail::level lev, typename Dummy>
    struct inner {
      static Scalar* execute(committed_descriptor& desc);
    };
  };

  /**
   * Calculates twiddle factors for the implementation in use.
   *
   * @return Scalar* USM pointer to the twiddle factors
   */
  Scalar* calculate_twiddles() { return dispatch<calculate_twiddles_struct>(); }

  /**
   * Builds the kernel bundle with appropriate values of specialization constants for the first supported subgroup size.
   *
   * @tparam SubgroupSize first subgroup size
   * @tparam OtherSGSizes other subgroup sizes
   * @return sycl::kernel_bundle<sycl::bundle_state::executable>
   */
  template <int SubgroupSize, int... OtherSGSizes>
  sycl::kernel_bundle<sycl::bundle_state::executable> build_w_spec_const() {
    // This function is called from constructor initializer list and it accesses other data members of the class. These
    // are already initialized by the time this is called only if they are declared in the class definition before the
    // member that is initialized by this function.
    if (std::count(supported_sg_sizes.begin(), supported_sg_sizes.end(), SubgroupSize)) {
      std::vector<sycl::kernel_id> ids;
      level = prepare_implementation<SubgroupSize>(ids);

      if (sycl::is_compatible(ids, dev)) {
        auto in_bundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(queue.get_context(), ids);
        set_spec_constants(in_bundle);
        used_sg_size = SubgroupSize;
        try {
          return sycl::build(in_bundle);
        } catch (std::exception& e) {
          std::cerr << "Build for subgroup size " << SubgroupSize << " failed with message:\n" << e.what() << std::endl;
        }
      }
    }
    if constexpr (sizeof...(OtherSGSizes) == 0) {
      throw unsupported_configuration("None of the compiled subgroup sizes are supported by the device!");
    } else {
      return build_w_spec_const<OtherSGSizes...>();
    }
  }

  /**
   * Constructor.
   *
   * @param params descriptor this is created from
   * @param queue queue to use when enqueueing device work
   */
  committed_descriptor(const descriptor<Scalar, Domain>& params, sycl::queue& queue)
      : params(params),
        queue(queue),
        dev(queue.get_device()),
        ctx(queue.get_context()),
        // get some properties we will use for tuning
        n_compute_units(dev.get_info<sycl::info::device::max_compute_units>()),
        supported_sg_sizes(dev.get_info<sycl::info::device::sub_group_sizes>()),
        num_sgs_per_wg(PORTFFT_SGS_IN_WG) {
    // Throw an exception if the descriptor is invalid.
    // One direction is allowed to be invalid but not both.
    params.validate(fwd_validation_err, bwd_validation_err);

    // compile the kernels
    exec_bundle = build_w_spec_const<PORTFFT_SUBGROUP_SIZES>();

    // get some properties we will use for tuning
    n_compute_units = dev.get_info<sycl::info::device::max_compute_units>();
    local_memory_size = queue.get_device().get_info<sycl::info::device::local_mem_size>();
    // TODO: Remove once we support global impl
    std::size_t minimum_local_mem_required =
        std::max(num_scalars_in_local_mem(direction::FORWARD), num_scalars_in_local_mem(direction::BACKWARD)) *
        sizeof(Scalar);
    if (minimum_local_mem_required > local_memory_size) {
      throw unsupported_configuration("Insufficient amount of local memory available: ", local_memory_size,
                                      "B. Required: ", minimum_local_mem_required, "B.");
    }
    twiddles_forward = std::shared_ptr<Scalar>(calculate_twiddles(), detail::free_usm(queue));
  }

 public:
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "Scalar must be either float or double!");
  /**
   * Alias for `Scalar`.
   */
  using scalar_type = Scalar;
  /**
   * Alias for `Domain`.
   */
  static constexpr domain domain_value = Domain;

  /**
   * Destructor
   */
  ~committed_descriptor() { queue.wait(); }

  /**
   * Get the original descriptor used to commit
   */
  const descriptor<Scalar, Domain>& get_descriptor() const { return params; }

  /**
   * Computes in-place forward FFT, working on a buffer.
   *
   * @param inout buffer containing input and output data
   */
  void compute_forward(sycl::buffer<complex_type, 1>& inout) {
    // For now we can just call out-of-place implementation.
    // This might need to be changed once we implement support for large sizes that work in global memory.
    compute_forward(inout, inout);
  }

  /**
   * Computes in-place backward FFT, working on a buffer.
   *
   * @param inout buffer containing input and output data
   */
  void compute_backward(sycl::buffer<complex_type, 1>& inout) {
    // For now we can just call out-of-place implementation.
    // This might need to be changed once we implement support for large sizes that work in global memory.
    compute_backward(inout, inout);
  }

  /**
   * Computes out-of-place forward FFT, working on buffers.
   * Input and output buffers must not overlap.
   *
   * @param in buffer containing input data
   * @param out buffer containing output data
   */
  void compute_forward(const sycl::buffer<complex_type, 1>& in, sycl::buffer<complex_type, 1>& out) {
    dispatch_kernel<direction::FORWARD>(in, out);
  }

  /**
   * Computes out-of-place forward FFT, working on buffers.
   * Input and output buffers must not overlap.
   *
   * @param in buffer containing input data
   * @param out buffer containing output data
   */
  void compute_forward(const sycl::buffer<Scalar, 1>& /*in*/, sycl::buffer<complex_type, 1>& /*out*/) {
    throw unsupported_configuration("Real to complex FFTs not yet implemented.");
  }

  /**
   * Compute out of place backward FFT, working on buffers
   * Input and output buffers must not overlap.
   *
   * @param in buffer containing input data
   * @param out buffer containing output data
   */
  void compute_backward(const sycl::buffer<complex_type, 1>& in, sycl::buffer<complex_type, 1>& out) {
    dispatch_kernel<direction::BACKWARD>(in, out);
  }

  /**
   * Computes in-place forward FFT, working on USM memory.
   *
   * @param inout USM pointer to memory containing input and output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_forward(complex_type* inout, const std::vector<sycl::event>& dependencies = {}) {
    // For now we can just call out-of-place implementation.
    // This might need to be changed once we implement support for large sizes that work in global memory.
    return compute_forward(inout, inout, dependencies);
  }

  /**
   * Computes in-place forward FFT, working on USM memory.
   *
   * @param inout USM pointer to memory containing input and output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_forward(Scalar* inout, const std::vector<sycl::event>& dependencies = {}) {
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
    return compute_backward(inout, inout, dependencies);
  }

  /**
   * Computes out-of-place forward FFT, working on USM memory.
   * Input and output pointers must not overlap.
   *
   * @param in USM pointer to memory containing input data
   * @param out USM pointer to memory containing output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_forward(const complex_type* in, complex_type* out,
                              const std::vector<sycl::event>& dependencies = {}) {
    return dispatch_kernel<direction::FORWARD>(in, out, dependencies);
  }

  /**
   * Computes out-of-place forward FFT, working on USM memory.
   * Input and output pointers must not overlap.
   *
   * @param in USM pointer to memory containing input data
   * @param out USM pointer to memory containing output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_forward(const Scalar* /*in*/, complex_type* /*out*/,
                              const std::vector<sycl::event>& /*dependencies*/ = {}) {
    throw unsupported_configuration("Real to complex FFTs not yet implemented.");
    return {};
  }

  /**
   * Computes out-of-place backward FFT, working on USM memory.
   * Input and output pointers must not overlap.
   *
   * @param in USM pointer to memory containing input data
   * @param out USM pointer to memory containing output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_backward(const complex_type* in, complex_type* out,
                               const std::vector<sycl::event>& dependencies = {}) {
    return dispatch_kernel<direction::BACKWARD>(in, out, dependencies);
  }

 private:
  /**
   * Dispatches the kernel with the first subgroup size that is supported by the device.
   *
   * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam T_in Type of the input buffer or USM pointer
   * @tparam T_out Type of the output buffer or USM pointer
   * @param in buffer or USM pointer to memory containing input data
   * @param out buffer or USM pointer to memory containing output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event
   */
  template <direction Dir, typename T_in, typename T_out>
  sycl::event dispatch_kernel(const T_in in, T_out out, const std::vector<sycl::event>& dependencies = {}) {
    if (Dir == direction::FORWARD && !fwd_validation_err.empty()) {
      throw invalid_configuration("Invalid forward FFT configuration. ", fwd_validation_err);
    } else if (Dir == direction::BACKWARD && !bwd_validation_err.empty()) {
      throw invalid_configuration("Invalid backward FFT configuration. ", bwd_validation_err);
    }
    return dispatch_kernel_helper<Dir, T_in, T_out, PORTFFT_SUBGROUP_SIZES>(in, out, dependencies);
  }

  /**
   * Helper for dispatching the kernel with the first subgroup size that is supported by the device.
   *
   * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam T_in Type of the input buffer or USM pointer
   * @tparam T_out Type of the output buffer or USM pointer
   * @tparam SubgroupSize first subgroup size
   * @tparam OtherSGSizes other subgroup sizes
   * @param in buffer or USM pointer to memory containing input data
   * @param out buffer or USM pointer to memory containing output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event
   */
  template <direction Dir, typename T_in, typename T_out, int SubgroupSize, int... OtherSGSizes>
  sycl::event dispatch_kernel_helper(const T_in in, T_out out, const std::vector<sycl::event>& dependencies = {}) {
    if (SubgroupSize == used_sg_size) {
      auto input_layout = detail::get_layout(params, Dir);
      auto output_layout = detail::get_layout(params, inv(Dir));
      if (input_layout == detail::layout::PACKED && output_layout == detail::layout::PACKED) {
        return run_kernel<Dir, detail::layout::PACKED, SubgroupSize>(in, out, dependencies);
      } else if (input_layout == detail::layout::BATCH_INTERLEAVED && output_layout == detail::layout::PACKED) {
        return run_kernel<Dir, detail::layout::BATCH_INTERLEAVED, SubgroupSize>(in, out, dependencies);
      } else {
        return run_kernel<Dir, detail::layout::UNPACKED, SubgroupSize>(in, out, dependencies);
      }
    }
    if constexpr (sizeof...(OtherSGSizes) == 0) {
      throw unsupported_configuration("None of the compiled subgroup sizes are supported by the device!");
    } else {
      return dispatch_kernel_helper<Dir, T_in, T_out, OtherSGSizes...>(in, out, dependencies);
    }
  }

  /**
   * Struct for dispatching `run_kernel()` call.
   *
   * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam LayoutIn Input layout. Output layout is assumed to be PACKED for now
   * @tparam SubgroupSize size of the subgroup
   * @tparam T_in Type of the input USM pointer or buffer
   * @tparam T_out Type of the output USM pointer or buffer
   */
  template <direction Dir, detail::layout LayoutIn, int SubgroupSize, typename T_in, typename T_out>
  struct run_kernel_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template <detail::level lev, typename Dummy>
    struct inner {
      static sycl::event execute(committed_descriptor& desc, const T_in& in, T_out& out,
                                 const std::vector<sycl::event>& dependencies);
    };
  };

  /**
   * Common interface to run the kernel called by compute_forward and compute_backward
   *
   * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam LayoutIn Input layout. Output layout is assumed to be PACKED for now.
   * @tparam SubgroupSize size of the subgroup
   * @tparam T_in Type of the input USM pointer or buffer
   * @tparam T_out Type of the output USM pointer or buffer
   * @param in USM pointer to memory containing input data
   * @param out USM pointer to memory containing output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event
   */
  template <direction Dir, detail::layout LayoutIn, int SubgroupSize, typename T_in, typename T_out>
  sycl::event run_kernel(const T_in& in, T_out& out, const std::vector<sycl::event>& dependencies) {
    return dispatch<run_kernel_struct<Dir, LayoutIn, SubgroupSize, T_in, T_out>>(in, out, dependencies);
  }
};  // class committed_descriptor

/**
 * A descriptor containing FFT problem parameters.
 *
 * @tparam Scalar type of the scalar used for computations
 * @tparam Domain domain of the FFT
 */
template <typename Scalar_, domain Domain_>
struct descriptor {
  /// Scalar type to determine the FFT precision
  using Scalar = Scalar_;
  static_assert(std::is_scalar_v<Scalar>, "Precision must be a scalar type");

  /** FFT domain.
   * Determines whether the input (resp. output) is real or complex in the forward (resp. backward) direction.
   */
  static constexpr domain Domain = Domain_;

  std::vector<std::size_t> lengths;
  Scalar forward_scale = 1;
  Scalar backward_scale = 1;
  std::size_t number_of_transforms;
  complex_storage complex_storage = complex_storage::COMPLEX;
  placement placement = placement::OUT_OF_PLACE;
  std::vector<std::size_t> forward_strides;
  std::vector<std::size_t> backward_strides;
  std::size_t forward_distance = 0;
  std::size_t backward_distance = 0;
  // TODO: add TRANSPOSE, WORKSPACE and ORDERING if we determine they make sense

  /**
   * Construct a new descriptor object.
   *
   * @param lengths size of the FFT transform
   * @param number_of_transforms if set, forward_distance and backward_distance will be set to default values so that
   * each FFT are continuous in memory.
   */
  explicit descriptor(std::vector<std::size_t> lengths, std::size_t number_of_transforms = 1)
      : lengths(lengths),
        number_of_transforms(number_of_transforms),
        forward_strides(lengths.size() + 1),
        backward_strides(lengths.size() + 1) {
    std::size_t total_length = get_flattened_length();
    backward_scale = Scalar(1) / static_cast<Scalar>(total_length);
    if (number_of_transforms > 1) {
      forward_distance = total_length;
      backward_distance = total_length;
    }
    std::size_t stride = 1;
    for (std::size_t i = lengths.size(); i > 0; --i) {
      forward_strides[i] = stride;
      backward_strides[i] = stride;
      stride *= lengths[i - 1];
    }
  }

  /**
   * Commits the descriptor, precalculating what can be done in advance.
   *
   * @param queue queue to use for computations
   * @return committed_descriptor<Scalar, Domain>
   */
  committed_descriptor<Scalar, Domain> commit(sycl::queue& queue) const { return {*this, queue}; }

  /**
   * Get the flattened length of an FFT for a single batch, ignoring strides and distance.
   */
  std::size_t get_flattened_length() const noexcept {
    return std::accumulate(lengths.begin(), lengths.end(), 1LU, std::multiplies<>());
  }

  /**
   * Get the number of elements required in the input buffer for a given direction.
   * Takes into account the lengths, number of transforms, strides (with offset) and direction.
   *
   * @param dir direction
   */
  std::size_t get_input_count(direction dir) const noexcept {
    return get_buffer_count(get_strides(dir), get_distance(dir));
  }

  /**
   * Get the number of elements required in the output buffer for a given direction.
   * Takes into account the lengths, number of transforms, strides (with offset) and direction.
   *
   * @param dir direction
   */
  std::size_t get_output_count(direction dir) const noexcept { return get_input_count(inv(dir)); }

  /**
   * Return the strides for a given direction
   *
   * @param dir direction
   */
  const std::vector<std::size_t>& get_strides(direction dir) const noexcept {
    return dir == direction::FORWARD ? forward_strides : backward_strides;
  }

  /**
   * Return the distance for a given direction
   *
   * @param dir direction
   */
  std::size_t get_distance(direction dir) const noexcept {
    return dir == direction::FORWARD ? forward_distance : backward_distance;
  }

  /**
   * Return the scale for a given direction
   *
   * @param dir direction
   */
  Scalar get_scale(direction dir) const noexcept { return dir == direction::FORWARD ? forward_scale : backward_scale; }

  /**
   * Check if the descriptor can be committed.
   * Throw @link invalid_configuration or @link unsupported_configuration if it cannot be committed.
   *
   * @param fwd_err Output string, empty if the forward direction is valid
   * @param bwd_err Output string, empty if the backward direction is valid
   */
  void validate(std::string& fwd_err, std::string& bwd_err) const {
    if (complex_storage != complex_storage::COMPLEX) {
      throw unsupported_configuration("portFFT only supports COMPLEX storage for now");
    }
    if (lengths.empty()) {
      throw invalid_configuration("Invalid lengths, must have at least 1 dimension");
    }
    if (lengths.size() != 1) {
      throw unsupported_configuration("portFFT only supports 1D FFT for now");
    }
    for (std::size_t i = 0; i < lengths.size(); ++i) {
      if (lengths[i] == 0) {
        throw invalid_configuration("Invalid lengths[", i, "]=", lengths[i], ", must be positive");
      }
    }
    if (number_of_transforms == 0) {
      throw invalid_configuration("Invalid number of transform ", number_of_transforms, ", must be positive");
    }
    validate_strides_distance(forward_strides, forward_distance, "forward");
    validate_strides_distance(backward_strides, backward_distance, "backward");
    validate_strides_distance_in_place();

    fwd_err = validate_overlap(direction::FORWARD);
    bwd_err = validate_overlap(direction::BACKWARD);
    if (!fwd_err.empty() && !bwd_err.empty()) {
      throw invalid_configuration("Invalid configuration for both directions.\nForward error:'", fwd_err,
                                  "'.\nBackward error:'", bwd_err, "'\n");
    }
  }

 private:
  /**
   * Throw an exception if the given stride and distance are invalid for any direction.
   *
   * @param strides strides to check
   * @param distance distance to check
   * @param dir_str direction string for errors
   */
  void validate_strides_distance(const std::vector<std::size_t>& strides, std::size_t distance,
                                 const std::string& dir_str) const {
    // Validate stride
    std::size_t expected_num_strides = lengths.size() + 1;
    if (strides.size() != expected_num_strides) {
      throw invalid_configuration("Mismatching ", dir_str, " strides length got ", strides.size(), " expected ",
                                  expected_num_strides);
    }
    for (std::size_t i = 1; i < strides.size(); ++i) {
      if (strides[i] == 0) {
        throw invalid_configuration("Invalid ", dir_str, " stride[", strides[i], "]=", strides[i],
                                    ", must be positive");
      }
    }

    // Validate distance
    if (number_of_transforms > 1 && distance == 0) {
      throw invalid_configuration("Invalid ", dir_str, " distance ", distance, ", must be positive for batched FFTs");
    }
  }

  /**
   * Require the same input and output strides and distance for in-place configurations.
   */
  void validate_strides_distance_in_place() const {
    if (placement != placement::IN_PLACE) {
      return;
    }

    if (forward_strides != backward_strides) {
      throw invalid_configuration("Invalid forward and backward strides must match for in-place configurations");
    }

    if (forward_distance != backward_distance) {
      throw invalid_configuration("Invalid forward and backward distances must match for in-place configurations");
    }
  }

  /**
   * Check that out-of-place FFTs don't overlap.
   * Two input indices must not write to the same output index.
   * Only supports 1D C2C transforms for now.
   *
   * @param dir Direction
   * @return An empty string if the FFT is valid for the given direction. Otherwise returns an error message.
   */
  std::string validate_overlap(direction dir) const {
    // TODO: Add support for R2C transforms
    if (Domain == domain::REAL) {
      return "unsupported";
    }

    const auto& output_strides = get_strides(inv(dir));
    const std::size_t output_distance = get_distance(inv(dir));

    // Quick check for most common configurations.
    // This check has some false-negative for some impractical configurations, see ArbitraryInterleavedTest.
    // View the output data as a N+1 dimensional tensor for a N-dimension FFT: the number of batch is just another
    // dimension with a stride of 'distance'. This sorts the dimensions from fastest moving (inner-most) to slowest
    // moving (outer-most) and check that the stride of a dimension is large enough to avoid overlapping the previous
    // dimension.
    std::vector<std::size_t> generic_output_strides(output_strides.begin() + 1, output_strides.end());
    std::vector<std::size_t> generic_output_sizes = lengths;
    if (number_of_transforms > 1) {
      generic_output_strides.push_back(output_distance);
      generic_output_sizes.push_back(number_of_transforms);
    }
    std::vector<std::size_t> indices(generic_output_sizes.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](std::size_t a, std::size_t b) { return generic_output_strides[a] < generic_output_strides[b]; });
    bool generic_valid = true;
    for (std::size_t i = 1; i < indices.size(); ++i) {
      generic_valid = generic_valid && generic_output_strides[indices[i - 1]] * generic_output_sizes[indices[i - 1]] <=
                                           generic_output_strides[indices[i]];
    }
    if (generic_valid) {
      return "";
    }

    // Arbitrary interleaved configurations are not supported for multiple-dimensions.
    if (lengths.size() != 1) {
      return "unsupported";
    }

    // Strides or distance differ for the input and output.
    // Compute the output indices for multiple batches and input index and work backward to determine if another batch
    // and input index will write to the same location.
    const auto& input_strides = get_strides(dir);
    const std::size_t input_distance = get_distance(dir);
    std::size_t fft_size = lengths[0];
    std::size_t input_offset = input_strides[0];
    std::size_t input_stride = input_strides[1];
    std::size_t output_offset = output_strides[0];
    std::size_t output_stride = output_strides[1];
    for (std::size_t b = 0; b < number_of_transforms; ++b) {
      for (std::size_t i = 0; i < fft_size; ++i) {
        std::size_t linear_input_idx = input_offset + b * input_distance + i * input_stride;
        std::size_t linear_output_idx = output_offset + b * output_distance + i * output_stride;
        // Check if another batch will write to the same output index.
        for (std::size_t other_b = b + 1; other_b < number_of_transforms; ++other_b) {
          std::size_t other_linear_output_idx = output_offset + other_b * output_distance;
          std::size_t diff = other_linear_output_idx > linear_output_idx ? other_linear_output_idx - linear_output_idx
                                                                         : linear_output_idx - other_linear_output_idx;
          if (diff % output_stride == 0) {
            std::size_t other_i = diff / output_stride;
            if (other_i < fft_size) {
              std::size_t other_linear_input_idx = input_offset + other_b * input_distance + other_i * input_stride;
              std::stringstream ss;
              ss << "Found overlapping output for batch=" << b << " index=" << i
                 << " (linear index=" << linear_input_idx << ") and other batch=" << other_b
                 << " other index=" << other_i << " (other linear index=" << other_linear_input_idx
                 << ") both writing at the output linear index=" << linear_output_idx;
              return ss.str();
            }
          }
        }
      }  // end of loop on input indices
    }    // end of loop on input batches
    return "";
  }  // end of validate_overlap

  /**
   * Compute the number of elements required for a buffer with the descriptor's length, number of transforms and the
   * given strides (with offset) and distance.
   *
   * @param strides buffer's strides
   * @param distance buffer's distance
   */
  std::size_t get_buffer_count(const std::vector<std::size_t>& strides, std::size_t distance) const noexcept {
    // Compute the last element that can be accessed
    std::size_t last_elt_idx = strides[0];  // offset
    last_elt_idx += (number_of_transforms - 1) * distance;
    for (std::size_t i = 0; i < lengths.size(); ++i) {
      last_elt_idx += (lengths[i] - 1) * strides[i + 1];
    }
    return last_elt_idx + 1;
  }
};

}  // namespace portfft

#endif
