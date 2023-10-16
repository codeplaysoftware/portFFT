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
#include <defines.hpp>
#include <enums.hpp>
#include <utils.hpp>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>
#include <utility>

namespace portfft {

namespace detail {

// kernel names
// TODO: Remove all templates except Scalar, Domain and Memory and SubgroupSize
template <typename Scalar, domain, direction, detail::memory, detail::layout, detail::layout,
          detail::elementwise_multiply, detail::elementwise_multiply, detail::apply_scale_factor, Idx SubgroupSize>
class workitem_kernel;
template <typename Scalar, domain, direction, detail::memory, detail::layout, detail::layout,
          detail::elementwise_multiply, detail::elementwise_multiply, detail::apply_scale_factor, Idx SubgroupSize>
class subgroup_kernel;
template <typename Scalar, domain, direction, detail::memory, detail::layout, detail::layout,
          detail::elementwise_multiply, detail::elementwise_multiply, detail::apply_scale_factor, Idx SubgroupSize>
class workgroup_kernel;

/**
 * Return whether the given descriptor has default strides and distance for a given direction.
 * Used internally only.
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

/**
 * @brief Get the layout described by the given descriptor.
 * The concept of layout is used internally only.
 * 
 * @tparam Descriptor Descriptor type
 * @param desc Given descriptor
 * @param dir Direction
 */
template <typename Descriptor>
layout get_layout(const Descriptor& desc, direction dir) {
  if (has_default_strides_and_distance(desc, dir)) {
    return layout::PACKED;
  }
  const auto& strides = desc.get_strides(dir);
  auto distance = desc.get_distance(dir);
  if (desc.number_of_transforms > 1 && distance == 1 && strides.size() == 1 && strides[0] == desc.number_of_transforms) {
    return layout::BATCH_INTERLEAVED;
  }
  return layout::UNPACKED;
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
  Idx n_compute_units;
  Idx local_memory_size;
  std::vector<std::size_t> supported_sg_sizes;
  Idx used_sg_size;
  Idx num_sgs_per_wg;
  std::shared_ptr<Scalar> twiddles_forward;
  detail::level level;
  std::vector<Idx> factors;
  sycl::kernel_bundle<sycl::bundle_state::executable> exec_bundle;
  // Non empty if the descriptor is not valid in fwd or bwd directions
  std::pair<std::string, std::string> fwd_bwd_validation_errs;

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
        throw unsupported_configuration("Unimplemented!");
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
        throw unsupported_configuration("Unimplemented!");
    }
  }

  /**
   * Prepares the implementation for the particular problem size. That includes factorizing it and getting ids for the
   * set of kernels that need to be JIT compiled.
   *
   * @tparam SubgroupSize size of the subgroup
   * @param[out] ids list of kernel ids that need to be JIT compiled
   * @return detail::level
   */
  template <Idx SubgroupSize>
  detail::level prepare_implementation(std::vector<sycl::kernel_id>& ids) {
    factors.clear();

    // TODO: check and support all the parameter values
    if constexpr (Domain != domain::COMPLEX) {
      throw unsupported_configuration("portFFT only supports complex to complex transforms");
    }
    if (params.lengths.size() != 1) {
      throw unsupported_configuration("portFFT only supports 1D FFT for now");
    }
    IdxGlobal fft_size = static_cast<IdxGlobal>(params.lengths[0]);
    if (!detail::cooley_tukey_size_list_t::has_size(fft_size)) {
      throw unsupported_configuration("FFT size ", fft_size, " is not compiled in!");
    }

    if (detail::fits_in_wi<Scalar>(fft_size)) {
      detail::get_ids<detail::workitem_kernel, Scalar, Domain, SubgroupSize>(ids);
      return detail::level::WORKITEM;
    }
    if (detail::fits_in_sg<Scalar>(fft_size, SubgroupSize)) {
      Idx factor_sg = detail::factorize_sg(static_cast<Idx>(fft_size), SubgroupSize);
      Idx factor_wi = static_cast<Idx>(fft_size) / factor_sg;
      // This factorization is duplicated in the dispatch logic on the device.
      // The CT and spec constant factors should match.
      factors.push_back(factor_wi);
      factors.push_back(factor_sg);
      detail::get_ids<detail::subgroup_kernel, Scalar, Domain, SubgroupSize>(ids);
      return detail::level::SUBGROUP;
    }
    IdxGlobal n_idx_global = detail::factorize(fft_size);
    Idx n = static_cast<Idx>(n_idx_global);
    Idx m = static_cast<Idx>(fft_size / n_idx_global);
    Idx factor_sg_n = detail::factorize_sg(n, SubgroupSize);
    Idx factor_wi_n = n / factor_sg_n;
    Idx factor_sg_m = detail::factorize_sg(m, SubgroupSize);
    Idx factor_wi_m = m / factor_sg_m;
    if (detail::fits_in_wi<Scalar>(factor_wi_n) && detail::fits_in_wi<Scalar>(factor_wi_m)) {
      factors.push_back(factor_wi_n);
      factors.push_back(factor_sg_n);
      factors.push_back(factor_wi_m);
      factors.push_back(factor_sg_m);
      // This factorization of N and M is duplicated in the dispatch logic on the device.
      // The CT and spec constant factors should match.
      detail::get_ids<detail::workgroup_kernel, Scalar, Domain, SubgroupSize>(ids);
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
    template <detail::level Lev, typename Dummy>
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
    template <detail::level Lev, typename Dummy>
    struct inner {
      static std::size_t execute(committed_descriptor& desc, direction dir);
    };
  };

  /**
   * Determine the number of scalars we need to have space for in the local memory. It may also modify `num_sgs_in_wg`
   * to make the problem fit in the local memory.
   *
   * @param dir Direction of the FFT is used to determine the input layout of the descriptor which can affect the local memory size.
   * @return the number of scalars
   */
  std::size_t num_scalars_in_local_mem(direction dir) {
    return dispatch<num_scalars_in_local_mem_struct>(dir);
  }

  /**
   * Struct for dispatching `calculate_twiddles()` call.
   */
  struct calculate_twiddles_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template <detail::level Lev, typename Dummy>
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
  template <Idx SubgroupSize, Idx... OtherSGSizes>
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
      throw invalid_configuration("None of the compiled subgroup sizes are supported by the device!");
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
        n_compute_units(static_cast<Idx>(dev.get_info<sycl::info::device::max_compute_units>())),
        local_memory_size(static_cast<Idx>(dev.get_info<sycl::info::device::local_mem_size>())),
        supported_sg_sizes(dev.get_info<sycl::info::device::sub_group_sizes>()),
        used_sg_size(0),
        num_sgs_per_wg(PORTFFT_SGS_IN_WG),
        // validate the descriptor
        fwd_bwd_validation_errs(params.validate()),
        // compile the kernels
        exec_bundle(build_w_spec_const<PORTFFT_SUBGROUP_SIZES>()) {
    twiddles_forward = std::shared_ptr<Scalar>(calculate_twiddles(), [queue](Scalar* ptr) {
      if (ptr != nullptr) {
        sycl::free(ptr, queue);
      }
    });
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
  static constexpr domain DomainValue = Domain;

  /**
   * Destructor
   */
  ~committed_descriptor() { queue.wait(); }

  // rule of three
  committed_descriptor(const committed_descriptor& other) = default;
  committed_descriptor& operator=(const committed_descriptor& other) = default;

  // default construction is not appropriate
  committed_descriptor() = delete;

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
   * Compute out-of-place backward FFT, working on buffers.
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
   * @tparam TIn Type of the input buffer or USM pointer
   * @tparam TOut Type of the output buffer or USM pointer
   * @param in buffer or USM pointer to memory containing input data
   * @param out buffer or USM pointer to memory containing output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event
   */
  template <direction Dir, typename TIn, typename TOut>
  sycl::event dispatch_kernel(const TIn in, TOut out, const std::vector<sycl::event>& dependencies = {}) {
    if (Dir == direction::FORWARD && !fwd_bwd_validation_errs.first.empty()) {
      throw invalid_configuration("Invalid forward FFT configuration. ", fwd_bwd_validation_errs.first);
    } else if (Dir == direction::BACKWARD && !fwd_bwd_validation_errs.second.empty()) {
      throw invalid_configuration("Invalid backward FFT configuration. ", fwd_bwd_validation_errs.second);
    }
    return dispatch_kernel_helper<Dir, TIn, TOut, PORTFFT_SUBGROUP_SIZES>(in, out, dependencies);
  }

  /**
   * Helper for dispatching the kernel with the first subgroup size that is supported by the device.
   *
   * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam TIn Type of the input buffer or USM pointer
   * @tparam TOut Type of the output buffer or USM pointer
   * @tparam SubgroupSize first subgroup size
   * @tparam OtherSGSizes other subgroup sizes
   * @param in buffer or USM pointer to memory containing input data
   * @param out buffer or USM pointer to memory containing output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event
   */
  template <direction Dir, typename TIn, typename TOut, Idx SubgroupSize, Idx... OtherSGSizes>
  sycl::event dispatch_kernel_helper(const TIn in, TOut out, const std::vector<sycl::event>& dependencies = {}) {
    if (SubgroupSize == used_sg_size) {
      std::size_t minimum_local_mem_required = num_scalars_in_local_mem(Dir) * sizeof(Scalar);
      if (static_cast<Idx>(minimum_local_mem_required) > local_memory_size) {
        throw out_of_local_memory_error(
            "Insufficient amount of local memory available: " + std::to_string(local_memory_size) +
            "B. Required: " + std::to_string(minimum_local_mem_required) + "B.");
      }
      constexpr auto PACKED = detail::layout::PACKED;
      constexpr auto UNPACKED = detail::layout::UNPACKED;
      constexpr auto BATCH_INTERLEAVED = detail::layout::BATCH_INTERLEAVED;
      auto input_layout = detail::get_layout(params, Dir);
      auto output_layout = detail::get_layout(params, inv(Dir));
      auto scale_factor = params.get_scale(Dir);
      if (input_layout == PACKED && output_layout == PACKED) {
        return run_kernel<Dir, PACKED, PACKED, SubgroupSize>(in, out, scale_factor,
                                                                                             dependencies);
      }
      if (input_layout == BATCH_INTERLEAVED && output_layout == PACKED) {
        return run_kernel<Dir, BATCH_INTERLEAVED, PACKED, SubgroupSize>(
            in, out, scale_factor, dependencies);
      }
      if (input_layout == PACKED && output_layout == BATCH_INTERLEAVED) {
        return run_kernel<Dir, PACKED, BATCH_INTERLEAVED, SubgroupSize>(
            in, out, scale_factor, dependencies);
      }
      if (input_layout == BATCH_INTERLEAVED && output_layout == BATCH_INTERLEAVED) {
        return run_kernel<Dir, BATCH_INTERLEAVED, BATCH_INTERLEAVED, SubgroupSize>(
            in, out, scale_factor, dependencies);
      }
      // Treat all other configurations as non-optimized unpacked
        return run_kernel<Dir, UNPACKED, UNPACKED, SubgroupSize>(
            in, out, scale_factor, dependencies);
    }
    if constexpr (sizeof...(OtherSGSizes) == 0) {
      throw invalid_configuration("None of the compiled subgroup sizes are supported by the device!");
    } else {
      return dispatch_kernel_helper<Dir, TIn, TOut, OtherSGSizes...>(in, out, dependencies);
    }
  }

  /**
   * Struct for dispatching `run_kernel()` call.
   *
   * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam LayoutIn Input Layout
   * @tparam LayoutOut Output Layout
   * @tparam SubgroupSize size of the subgroup
   * @tparam TIn Type of the input USM pointer or buffer
   * @tparam TOut Type of the output USM pointer or buffer
   */
  template <direction Dir, detail::layout LayoutIn, detail::layout LayoutOut, Idx SubgroupSize, typename TIn,
            typename TOut>
  struct run_kernel_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template <detail::level Lev, typename Dummy>
    struct inner {
      static sycl::event execute(committed_descriptor& desc, const TIn& in, TOut& out, Scalar scale_factor,
                                 const std::vector<sycl::event>& dependencies);
    };
  };

  /**
   * Common interface to run the kernel called by compute_forward and compute_backward
   *
   * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam LayoutIn Input Layout
   * @tparam LayoutOut Output Layout
   * @tparam SubgroupSize size of the subgroup
   * @tparam TIn Type of the input USM pointer or buffer
   * @tparam TOut Type of the output USM pointer or buffer
   * @param in USM pointer to memory containing input data
   * @param out USM pointer to memory containing output data
   * @param scale_factor Value with which the result of the FFT will be multiplied
   * @param dependencies events that must complete before the computation
   * @return sycl::event
   */
  template <direction Dir, detail::layout LayoutIn, detail::layout LayoutOut, Idx SubgroupSize, typename TIn,
            typename TOut>
  sycl::event run_kernel(const TIn& in, TOut& out, Scalar scale_factor, const std::vector<sycl::event>& dependencies) {
    return dispatch<run_kernel_struct<Dir, LayoutIn, LayoutOut, SubgroupSize, TIn, TOut>>(in, out, scale_factor,
                                                                                          dependencies);
  }
};

#undef PORTFFT_DISPATCH

/**
 * A descriptor containing FFT problem parameters.
 *
 * @tparam DescScalar type of the scalar used for computations
 * @tparam DescDomain domain of the FFT
 */
template <typename DescScalar, domain DescDomain>
struct descriptor {
  /// Scalar type to determine the FFT precision.
  using Scalar = DescScalar;
  static_assert(std::is_floating_point_v<Scalar>, "Precision must be a scalar type");

  /**
   * FFT domain.
   * Determines whether the input (resp. output) is real or complex in the forward (resp. backward) direction.
   */
  static constexpr domain Domain = DescDomain;

  /**
   * The lengths in elements of each dimension. Only 1D transforms are supported. Must be specified.
   */
  std::vector<std::size_t> lengths;
  /**
   * A scaling factor applied to the output of forward transforms. Default value is 1.
   */
  Scalar forward_scale = 1;
  /**
   * A scaling factor applied to the output of backward transforms. Default value is the reciprocal of the
   * product of the lengths.
   * NB a forward transform followed by a backward transform with both forward_scale and
   * backward_scale set to 1 will result in the data being scaled by the product of the lengths.
   */
  Scalar backward_scale = 1;
  /**
   * The number of transforms or batches that will be solved with each call to compute_xxxward.
   */
  std::size_t number_of_transforms;
  /**
   * The data layout of complex values. Default value is complex_storage::COMPLEX. complex_storage::COMPLEX
   * indicates that the real and imaginary part of a complex number is contiguous i.e an Array of Structures.
   * complex_storage::REAL_REAL indicates that all the real values are contiguous and all the imaginary values are
   * contiguous i.e. a Structure of Arrays. Only complex_storage::COMPLEX is supported.
   */
  complex_storage complex_storage = complex_storage::COMPLEX;
  /**
   * Indicates if the memory address of the output pointer is the same as the input pointer. Default value is
   * placement::OUT_OF_PLACE. When placement::OUT_OF_PLACE is used, only the out of place compute_xxxward functions can
   * be used and the memory pointed to by the input pointer and the memory pointed to by the output pointer must not
   * overlap at all. When placement::IN_PLACE is used, only the in-place compute_xxxward functions can be used.
   */
  placement placement = placement::OUT_OF_PLACE;
  /**
   * The strides of the data in the forward domain in elements.
   * Strides do not include the offset.
   */
  std::vector<std::size_t> forward_strides;
  /**
   * The strides of the data in the backward domain in elements.
   * Strides do not include the offset.
   */
  std::vector<std::size_t> backward_strides;
  /**
   * The number of elements between the first value of each transform in the forward domain. Must be set if \p number_of_transforms is greater than 1.
   */
  std::size_t forward_distance = 0;
  /**
   * The number of elements between the first value of each transform in the backward domain. Must be set if \p number_of_transforms is greater than 1. 
   */
  std::size_t backward_distance = 0;
  // TODO: add TRANSPOSE, WORKSPACE and ORDERING if we determine they make sense

  /**
   * Construct a new descriptor object.
   *
   * @param lengths size of the FFT transform. Default values for the forward and backward strides are set based on the lengths so that each FFT is contiguous in memory.
   * @param number_of_transforms Number of FFT transforms, also called batch size. Defaults to 1. If number_of_transforms is greater than 1 this also automatically sets the forward and backward distances to a default value so that each FFTs are stored one after the other.
   */
  explicit descriptor(std::vector<std::size_t> lengths, std::size_t number_of_transforms = 1)
      : lengths(lengths),
        number_of_transforms(number_of_transforms),
        forward_strides(lengths.size()),
        backward_strides(lengths.size()) {
    std::size_t total_length = get_flattened_length();
    backward_scale = Scalar(1) / static_cast<Scalar>(total_length);
    if (number_of_transforms > 1) {
      forward_distance = total_length;
      backward_distance = total_length;
    }
    std::size_t stride = 1;
    for (int i = static_cast<int>(lengths.size()) - 1; i >= 0; --i) {
      forward_strides[i] = stride;
      backward_strides[i] = stride;
      stride *= lengths[i];
    }
  }

  /**
   * Commits the descriptor, precalculating what can be done in advance.
   *
   * @param queue queue to use for computations
   * @return committed_descriptor<Scalar, Domain>
   */
  committed_descriptor<Scalar, Domain> commit(sycl::queue& queue) { return {*this, queue}; }

  /**
   * Get the flattened length of an FFT for a single batch, ignoring strides and distance.
   */
  std::size_t get_flattened_length() const noexcept {
    return std::accumulate(lengths.begin(), lengths.end(), 1LU, std::multiplies<std::size_t>());
  }

  /**
   * Get the size of the input buffer for a given direction in terms of the number of elements.
   * The number of elements is the same irrespective of the FFT domain.
   * Takes into account the lengths, number of transforms, strides and direction.
   *
   * @param dir direction
   */
  std::size_t get_input_count(direction dir) const noexcept {
    return get_buffer_count(get_strides(dir), get_distance(dir));
  }

  /**
   * Get the size of the output buffer for a given direction in terms of the number of elements.
   * The number of elements is the same irrespective of the FFT domain.
   * Takes into account the lengths, number of transforms, strides and direction.
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
   * Return a mutable reference to the strides for a given direction
   *
   * @param dir direction
   */
  std::vector<std::size_t>& get_strides(direction dir) noexcept {
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
   * Return a mutable reference to the distance for a given direction
   *
   * @param dir direction
   */
  std::size_t& get_distance(direction dir) noexcept {
    return dir == direction::FORWARD ? forward_distance : backward_distance;
  }

  /**
   * Return the scale for a given direction
   *
   * @param dir direction
   */
  Scalar get_scale(direction dir) const noexcept { return dir == direction::FORWARD ? forward_scale : backward_scale; }

  /**
   * Return a mutable reference to the scale for a given direction
   *
   * @param dir direction
   */
  Scalar& get_scale(direction dir) noexcept { return dir == direction::FORWARD ? forward_scale : backward_scale; }

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
    std::size_t expected_num_strides = lengths.size();
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
      return "REAL domain is unsupported";
    }

    const auto& output_strides = get_strides(inv(dir));
    const std::size_t output_distance = get_distance(inv(dir));

    // Quick check for most common configurations.
    // This check has some false-negative for some impractical configurations, see ArbitraryInterleavedTest.
    // View the output data as a N+1 dimensional tensor for a N-dimension FFT: the number of batch is just another
    // dimension with a stride of 'distance'. This sorts the dimensions from fastest moving (inner-most) to slowest
    // moving (outer-most) and check that the stride of a dimension is large enough to avoid overlapping the previous
    // dimension.
    std::vector<std::size_t> generic_output_strides(output_strides);
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
      return "multi-dim FFTs are unsupported";
    }

    // Strides or distance differ for the input and output.
    // Compute the output indices for multiple batches and input index and work backward to determine if another batch
    // and input index will write to the same location.
    // TODO: Test for input_offset and output_offset > 0
    const auto& input_strides = get_strides(dir);
    const std::size_t input_distance = get_distance(dir);
    std::size_t fft_size = lengths[0];
    std::size_t input_offset = 0;
    std::size_t input_stride = input_strides[0];
    std::size_t output_offset = 0;
    std::size_t output_stride = output_strides[0];
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
   * given strides and distance.
   * The number of elements is the same irrespective of the FFT domain.
   *
   * @param strides buffer's strides
   * @param distance buffer's distance
   */
  std::size_t get_buffer_count(const std::vector<std::size_t>& strides, std::size_t distance) const noexcept {
    // Compute the last element that can be accessed
    // TODO: Take into account offset
    std::size_t last_elt_idx = (number_of_transforms - 1) * distance;
    for (std::size_t i = 0; i < lengths.size(); ++i) {
      last_elt_idx += (lengths[i] - 1) * strides[i];
    }
    return last_elt_idx + 1;
  }
};

}  // namespace portfft

#endif
