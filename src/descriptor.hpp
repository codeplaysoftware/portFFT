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

#include <complex>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

namespace portfft {

namespace detail {

// kernel names
template <typename Scalar, domain Domain, direction Dir, detail::memory, detail::layout LayoutIn,
          Idx SubgroupSize>
class workitem_kernel;
template <typename Scalar, domain Domain, direction Dir, detail::memory, detail::layout LayoutIn,
          Idx SubgroupSize>
class subgroup_kernel;
template <typename Scalar, domain Domain, direction Dir, detail::memory, detail::layout LayoutIn,
          Idx SubgroupSize>
class workgroup_kernel;

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
  std::vector<std::size_t> supported_sg_sizes;
  Idx used_sg_size;
  std::shared_ptr<Scalar> twiddles_forward;
  detail::level level;
  std::vector<Idx> factors;
  sycl::kernel_bundle<sycl::bundle_state::executable> exec_bundle;
  Idx num_sgs_per_wg;
  Idx local_memory_size;

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
   * Get kernel ids for the implementation used.
   *
   * @tparam kernel which base template for kernel to use
   * @tparam SubgroupSize size of the subgroup
   * @param ids vector of kernel ids
   */
  template <template <typename, domain, direction, detail::memory, detail::layout, Idx> class Kernel,
            Idx SubgroupSize>
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
      get_ids<detail::workitem_kernel, SubgroupSize>(ids);
      return detail::level::WORKITEM;
    }
    if (detail::fits_in_sg<Scalar>(fft_size, SubgroupSize)) {
      Idx factor_sg = detail::factorize_sg(static_cast<Idx>(fft_size), SubgroupSize);
      Idx factor_wi = static_cast<Idx>(fft_size) / factor_sg;
      // This factorization is duplicated in the dispatch logic on the device.
      // The CT and spec constant factors should match.
      factors.push_back(factor_wi);
      factors.push_back(factor_sg);
      get_ids<detail::subgroup_kernel, SubgroupSize>(ids);
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
    template <detail::level Lev, detail::layout LayoutIn, typename Dummy>
    struct inner {
      static std::size_t execute(committed_descriptor& desc);
    };
  };

  /**
   * Determine the number of scalars we need to have space for in the local memory. It may also modify `num_sgs_in_wg`
   * to make the problem fit in the local memory.
   *
   * @return the number of scalars
   */
  template <detail::layout LayoutIn>
  std::size_t num_scalars_in_local_mem() {
    return dispatch<num_scalars_in_local_mem_struct, LayoutIn>();
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
        // get some properties we will use for tunning
        n_compute_units(static_cast<Idx>(dev.get_info<sycl::info::device::max_compute_units>())),
        supported_sg_sizes(dev.get_info<sycl::info::device::sub_group_sizes>()),
        // compile the kernels
        exec_bundle(build_w_spec_const<PORTFFT_SUBGROUP_SIZES>()),
        num_sgs_per_wg(PORTFFT_SGS_IN_WG) {
    // get some properties we will use for tuning
    local_memory_size = static_cast<Idx>(queue.get_device().get_info<sycl::info::device::local_mem_size>());
    std::size_t minimum_local_mem_required;
    if (params.forward_distance == 1 || params.backward_distance == 1) {
      minimum_local_mem_required = num_scalars_in_local_mem<detail::layout::BATCH_INTERLEAVED>() * sizeof(Scalar);
    } else {
      minimum_local_mem_required = num_scalars_in_local_mem<detail::layout::PACKED>() * sizeof(Scalar);
    }
    if (static_cast<Idx>(minimum_local_mem_required) > local_memory_size) {
      throw unsupported_configuration("Insufficient amount of local memory available: ", local_memory_size,
                                      "B. Required: ", minimum_local_mem_required, "B.");
    }
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
   *
   * @param in buffer containing input data
   * @param out buffer containing output data
   */
  void compute_forward(const sycl::buffer<complex_type, 1>& in, sycl::buffer<complex_type, 1>& out) {
    dispatch_kernel<direction::FORWARD>(in, out);
  }

  /**
   * Computes out-of-place forward FFT, working on buffers.
   *
   * @param in buffer containing input data
   * @param out buffer containing output data
   */
  void compute_forward(const sycl::buffer<Scalar, 1>& /*in*/, sycl::buffer<complex_type, 1>& /*out*/) {
    throw unsupported_configuration("Real to complex FFTs not yet implemented.");
  }

  /**
   * Compute out of place backward FFT, working on buffers
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
      std::size_t fft_size = params.lengths[0];  // 1d only for now
      std::size_t input_distance;
      std::size_t output_distance;
      Scalar scale_factor;
      if constexpr (Dir == direction::FORWARD) {
        input_distance = params.forward_distance;
        output_distance = params.backward_distance;
        scale_factor = params.forward_scale;
      } else {
        input_distance = params.backward_distance;
        output_distance = params.forward_distance;
        scale_factor = params.backward_scale;
      }
      if (input_distance == fft_size && output_distance == fft_size) {
        return run_kernel<Dir, detail::layout::PACKED, SubgroupSize>(in, out, scale_factor, dependencies);
      }
      if (input_distance == 1 && output_distance == fft_size && in != out) {
        return run_kernel<Dir, detail::layout::BATCH_INTERLEAVED, SubgroupSize>(in, out, scale_factor, dependencies);
      }
      throw unsupported_configuration("Only PACKED or BATCH_INTERLEAVED transforms are supported");
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
   * @tparam LayoutIn Input layout
   * @tparam SubgroupSize size of the subgroup
   * @tparam TIn Type of the input USM pointer or buffer
   * @tparam TOut Type of the output USM pointer or buffer
   */
  template <direction Dir, detail::layout LayoutIn, Idx SubgroupSize, typename TIn, typename TOut>
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
   * @tparam LayoutIn Input layout
   * @tparam SubgroupSize size of the subgroup
   * @tparam TIn Type of the input USM pointer or buffer
   * @tparam TOut Type of the output USM pointer or buffer
   * @param in USM pointer to memory containing input data
   * @param out USM pointer to memory containing output data
   * @param scale_factor Value with which the result of the FFT will be multiplied
   * @param dependencies events that must complete before the computation
   * @return sycl::event
   */
  template <direction Dir, detail::layout LayoutIn, Idx SubgroupSize, typename TIn, typename TOut>
  sycl::event run_kernel(const TIn& in, TOut& out, Scalar scale_factor, const std::vector<sycl::event>& dependencies) {
    return dispatch<run_kernel_struct<Dir, LayoutIn, SubgroupSize, TIn, TOut>>(in, out, scale_factor, dependencies);
  }
};

#undef PORTFFT_DISPATCH

/**
 * A descriptor containing FFT problem parameters.
 *
 * @tparam Scalar type of the scalar used for computations
 * @tparam Domain domain of the FFT
 */
template <typename Scalar, domain Domain>
struct descriptor {
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
   * The number of transforms or batches that will be solved with each call to compute_xxxward. Default value
   * is 1.
   */
  std::size_t number_of_transforms = 1;
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
   * The strides of the data in the forward domain in elements. The default value is {1}. Only {1} or
   * {number_of_transforms} is supported. Exactly one of `forward_strides` and `forward_distance` must be 1.
   */
  std::vector<std::size_t> forward_strides;
  /**
   * The strides of the data in the backward domain in elements. The default value is {1}. Must be the same as
   * forward_strides.
   */
  std::vector<std::size_t> backward_strides;
  /**
   * The number of elements between the first value of each transform in the forward domain. The default value is
   * lengths[0]. Must be either 1 or lengths[0]. Exactly one of `forward_strides` and `forward_distance` must be 1.
   */
  std::size_t forward_distance = 1;
  /**
   * The number of elements between the first value of each transform in the backward domain. The default value
   * is lengths[0]. Must be the same as forward_distance.
   */
  std::size_t backward_distance = 1;
  // TODO: add TRANSPOSE, WORKSPACE and ORDERING if we determine they make sense

  /**
   * Construct a new descriptor object.
   *
   * @param lengths size of the FFT transform
   */
  explicit descriptor(std::vector<std::size_t> lengths)
      : lengths(lengths),
        forward_strides{1},
        backward_strides{1},
        forward_distance(lengths[0]),
        backward_distance(lengths[0]) {
    // TODO: properly set default values for forward_strides, backward_strides, forward_distance, backward_distance
    for (auto l : lengths) {
      backward_scale *= Scalar(1) / static_cast<Scalar>(l);
    }
  }

  /**
   * Commits the descriptor, precalculating what can be done in advance.
   *
   * @param queue queue to use for computations
   * @return committed_descriptor<Scalar, Domain>
   */
  committed_descriptor<Scalar, Domain> commit(sycl::queue& queue) { return {*this, queue}; }

  std::size_t get_total_length() const noexcept {
    return std::accumulate(lengths.begin(), lengths.end(), 1LU, std::multiplies<std::size_t>());
  }
};

}  // namespace portfft

#endif
