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
 *  Codeplay's SYCL-FFT
 *
 **************************************************************************/

#ifndef SYCL_FFT_DESCRIPTOR_HPP
#define SYCL_FFT_DESCRIPTOR_HPP

#include <enums.hpp>
#include <common/subgroup.hpp>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

namespace sycl_fft {

namespace detail {

// kernel names
template <typename Scalar, domain Domain, direction Dir, detail::memory, detail::transpose TransposeIn, int SubgroupSize>
class workitem_kernel;
template <typename Scalar, domain Domain, direction Dir, detail::memory, detail::transpose TransposeIn, int SubgroupSize>
class subgroup_kernel;
template <typename Scalar, domain Domain, direction Dir, detail::memory, detail::transpose TransposeIn, int SubgroupSize>
class workgroup_kernel;

}  // namespace detail

// forward declaration
template <typename Scalar, domain Domain>
struct descriptor;

/*
Compute functions in the `committed_descriptor` call `dispatch_kernel` and `dispatch_kernel_helper`. These two functions
ensure the kernel is run with a supported subgroup size. Next `dispatch_kernel_helper` calls `run_kernel`. There are two
overloads of `run_kernel`, one for buffer interfaces and one for USM. `run_kernel` handles differences between forward
and backward computations, casts the memory (USM or buffers) from complex to scalars and launches the kernel.

Many of the parameters for the kernel, such as number of workitems launched and the required size of local allocations
are determined by the helpers `num_scalars_in_local_mem` and `get_global_size` from `dispatcher.hpp`. The kernel calls
`dispatcher`. From here on, each function has only one templated overload that handles both directions of transforms and
buffer and USM memory.

`dispatcher` determines which implementation to use for the particular FFT size and calls one of
the dispatcher functions for the particular implementation: `workitem_dispatcher` or `subgroup_dispatcher`. In case of
subgroup, it also factors the FFT size into one factor that fits into individual workitem and one that can be done
across workitems in a subgroup. `dispatcher` and all other device functions make no assumptions on the size of a work
group or the number of workgroups in a kernel. These numbers can be tuned for each device. TODO: currently we always
test one subgroup per workgroup, so more may or may not actually work correctly.

Both dispatcher functions are there to make the size of the FFT that is handled by the individual workitems compile time
constant. `subgroup_dispatcher` also calls `cross_sg_dispatcher` that makes the cross-subgroup factor of FFT size
compile time constant. They do that by using a switch on the FFT size for one workitem, before calling `workitem_impl`
or `subgroup_impl` respectively. The `_impl` functions take the FFT size for one workitem as a template parameter. Only
the calls that are determined to fit into available registers (depending on the value of SYCLFFT_TARGET_REGS_PER_WI
macro) are actually instantiated.

The `workitem_impl` and `subgroup_impl` functions iterate over the batch of problems, loading data for each first in
local memory then from there into private one. This is done in these two steps to avoid non-coalesced global memory
accesses. `workitem_impl` loads one problem per workitem and `subgroup_impl` loads one problem per subgroup. After doing
computations by the calls to `wi_dft` for workitem and `sg_dft` for subgroup the data is written out, going through
local memory again.

The computational parts of the implementations are further documented in files with their implementations `workitem.hpp`
and `subgroup.hpp`.
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
  std::size_t n_compute_units;
  std::vector<std::size_t> supported_sg_sizes;
  int used_sg_size;
  Scalar* twiddles_forward;
  detail::level level;
  std::vector<int> factors;
  sycl::kernel_bundle<sycl::bundle_state::executable> exec_bundle;

#define SYCL_FFT_DISPATCH(LEVEL, ...) \
switch (LEVEL) { \
case detail::level::WORKITEM: \
    return workitem_impl::__VA_ARGS__; \
case detail::level::SUBGROUP: \
    return subgroup_impl::__VA_ARGS__; \
case detail::level::WORKGROUP: \
    return workgroup_impl::__VA_ARGS__; \
default: \
    throw std::runtime_error("Unimplemented!"); \
}

#define SYCL_FFT_COMMITTED_DESCRIPTOR_DECLARE_IMPL(NAME) \
  struct NAME{ \
    /** \
     * Sets specialization constant values \
     * \
     * @param desc committed descriptor \
     * @param in_bundle buntdle to set specilaization constant values on \
     */ \
    static void set_spec_constants(committed_descriptor& desc, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle); \
    \
    /** \
     * Calculates the number of scalars for which space is needed in local memory. \
     * \
     * @param desc committed descriptor \
     * @return std::size_t number of scalars \
     */ \
    static std::size_t num_scalars_in_local_mem(committed_descriptor& desc); \
     \
    /** \
     * Calculates twiddle factors needed for given problem. \
     *  \
     * @param desc committed descriptor \
     * @return T* pointer to device memory containing twiddle factors \
     */ \
    static Scalar* calculate_twiddles(committed_descriptor& desc); \
    \
    /** \
     * Common interface to run the kernel called by compute_forward and compute_backward \
     * \
     * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD \
     * @tparam TransposeIn whether input is transposed (interpreting it as a matrix of batch size times FFT size) \
     * @tparam SubgroupSize size of the subgroup \
     * @tparam T_in Type of the input USM pointer or buffer \
     * @tparam T_out Type of the output USM pointer or buffer \
     * @param fft_size size of one FFT problem \
     * @param n_transforms number of FFT transforms to do in one call \
     * @param in USM pointer to memory containing input data \
     * @param out USM pointer to memory containing output data \
     * @param scale_factor Value with which the result of the FFT will be multiplied \
     * @param dependencies events that must complete before the computation \
     * @return sycl::event \
     */ \
    template <direction Dir, detail::transpose TransposeIn, int SubgroupSize, typename T_in, typename T_out> \
    static sycl::event run_kernel(committed_descriptor& desc, const T_in& in, T_out& out, Scalar scale_factor, const std::vector<sycl::event>& dependencies); \
  }

  SYCL_FFT_COMMITTED_DESCRIPTOR_DECLARE_IMPL(workitem_impl);
  SYCL_FFT_COMMITTED_DESCRIPTOR_DECLARE_IMPL(subgroup_impl);
  SYCL_FFT_COMMITTED_DESCRIPTOR_DECLARE_IMPL(workgroup_impl);

#undef SYCL_FFT_COMMITTED_DESCRIPTOR_DECLARE_IMPL

  /**
   * Get kernel ids for the implementation used.
   * 
   * @tparam kernel which base template for kernel to use
   * @tparam SubgroupSize size of the subgroup
   * @param ids vector of kernel ids
   */
  template<template<typename, domain, direction, detail::memory, detail::transpose, int> class Kernel, int SubgroupSize>
  void get_ids(std::vector<sycl::kernel_id>& ids){
      // if not used, some kernels might be optimized away in AOT compilation and not available here
      #define SYCL_FFT_GET_ID(DIRECTION,MEMORY,TRANSPOSE) \
      try { \
        ids.push_back(sycl::get_kernel_id<Kernel<Scalar, Domain, DIRECTION, MEMORY, TRANSPOSE, SubgroupSize>>()); \
      } catch (...) { \
      }

      SYCL_FFT_GET_ID(direction::FORWARD, detail::memory::BUFFER, detail::transpose::NOT_TRANSPOSED)
      SYCL_FFT_GET_ID(direction::BACKWARD, detail::memory::BUFFER, detail::transpose::NOT_TRANSPOSED)
      SYCL_FFT_GET_ID(direction::FORWARD, detail::memory::USM, detail::transpose::NOT_TRANSPOSED)
      SYCL_FFT_GET_ID(direction::BACKWARD, detail::memory::USM, detail::transpose::NOT_TRANSPOSED)
      SYCL_FFT_GET_ID(direction::FORWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED)
      SYCL_FFT_GET_ID(direction::BACKWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED)
      SYCL_FFT_GET_ID(direction::FORWARD, detail::memory::USM, detail::transpose::TRANSPOSED)
      SYCL_FFT_GET_ID(direction::BACKWARD, detail::memory::USM, detail::transpose::TRANSPOSED)

      #undef SYCL_FFT_GET_ID
  }

  /**
   * Prepares the implementation for the particular problem size. That includes factorizing it and getting ids for the set of kernels that need to be JIT compiled.
   * 
   * @tparam SubgroupSize size of the subgroup
   * @param[out] ids list of kernel ids that need to be JIT compiled
   * @return detail::level 
   */
  template <int SubgroupSize>
  detail::level prepare_implementation(std::vector<sycl::kernel_id>& ids){
    std::size_t fft_size = params.lengths[0];
    if (detail::fits_in_wi<Scalar>(fft_size)) {
      get_ids<detail::workitem_kernel, SubgroupSize>(ids);
      return detail::level::WORKITEM;
    }
    int factor_sg = detail::factorize_sg(static_cast<int>(fft_size), SubgroupSize);
    int factor_wi = static_cast<int>(fft_size) / factor_sg;
    if (detail::fits_in_wi<Scalar>(factor_wi)) {
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
      get_ids<detail::workgroup_kernel, SubgroupSize>(ids);
      return detail::level::WORKGROUP;
    }
    //TODO global
    throw std::runtime_error("FFT size " + std::to_string(N) + " is not supported!");
  }

  /**
   * Sets the implementation depentant specialization constant values.
   * 
   * @param in_bundle kernel bundle to set the specialization constants on
   */
  void set_spec_constants(sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle){
    SYCL_FFT_DISPATCH(level, set_spec_constants(*this, in_bundle))
  }

  /**
   * Determine the number of scalars we need to have space for in the local memory.
   * 
   * @return std::size_t the number of scalars
   */
  std::size_t num_scalars_in_local_mem(){
    SYCL_FFT_DISPATCH(level, num_scalars_in_local_mem(*this))
  }

  /**
   * Calculates twiddle factors for the implementation in use.
   * 
   * @return Scalar* USM pointer to the twiddle factors
   */
  Scalar* calculate_twiddles(){
    SYCL_FFT_DISPATCH(level, calculate_twiddles(*this))
  }

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
      throw std::runtime_error("None of the compiled subgroup sizes are supported by the device!");
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
        n_compute_units(dev.get_info<sycl::info::device::max_compute_units>()),
        supported_sg_sizes(dev.get_info<sycl::info::device::sub_group_sizes>()),
        // compile the kernels
        exec_bundle(build_w_spec_const<SYCLFFT_SUBGROUP_SIZES>()) {
    // TODO: check and support all the parameter values
    if (params.lengths.size() != 1) {
      throw std::runtime_error("SYCL-FFT only supports 1D FFT for now");
    }

    // get some properties we will use for tuning
    n_compute_units = dev.get_info<sycl::info::device::max_compute_units>();
    std::size_t local_memory_size = queue.get_device().get_info<sycl::info::device::local_mem_size>();
    std::size_t minimum_local_mem_required =  num_scalars_in_local_mem() * sizeof(Scalar);
    if (minimum_local_mem_required > local_memory_size) {
      throw std::runtime_error("Insufficient amount of local memory available: " + std::to_string(local_memory_size) +
                                "B. Required: " + std::to_string(minimum_local_mem_required) + "B.");
    }
    twiddles_forward = calculate_twiddles();
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
  ~committed_descriptor() {
    queue.wait();
    if (twiddles_forward != nullptr) {
      sycl::free(twiddles_forward, queue);
    }
  }

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

  committed_descriptor& operator=(const committed_descriptor&) = delete;
  committed_descriptor(const committed_descriptor&) = delete;

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
    return dispatch_kernel_helper<Dir, T_in, T_out, SYCLFFT_SUBGROUP_SIZES>(in, out, dependencies);
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
        return run_kernel<Dir, detail::transpose::NOT_TRANSPOSED, SubgroupSize>(in, out, scale_factor, dependencies);
      } else if (input_distance == 1 && output_distance == fft_size && in != out) {
        return run_kernel<Dir, detail::transpose::TRANSPOSED, SubgroupSize>(in, out, scale_factor, dependencies);
      } else {
        throw std::runtime_error("Unsupported configuration");
      }
    }
    if constexpr (sizeof...(OtherSGSizes) == 0) {
      throw std::runtime_error("None of the compiled subgroup sizes are supported by the device!");
    } else {
      return dispatch_kernel_helper<Dir, T_in, T_out, OtherSGSizes...>(in, out, dependencies);
    }
  }

  /**
   * Common interface to run the kernel called by compute_forward and compute_backward
   *
   * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam TransposeIn whether input is transposed (interpreting it as a matrix of batch size times FFT size)
   * @tparam SubgroupSize size of the subgroup
   * @tparam T_in Type of the input USM pointer or buffer
   * @tparam T_out Type of the output USM pointer or buffer
   * @param in USM pointer to memory containing input data
   * @param out USM pointer to memory containing output data
   * @param scale_factor Value with which the result of the FFT will be multiplied
   * @param dependencies events that must complete before the computation
   * @return sycl::event
   */
  template <direction Dir, detail::transpose TransposeIn, int SubgroupSize, typename T_in, typename T_out>
  sycl::event run_kernel(const T_in& in, T_out& out, Scalar scale_factor, const std::vector<sycl::event>& dependencies) {
    SYCL_FFT_DISPATCH(level, template run_kernel<Dir, TransposeIn, SubgroupSize>(*this, in, out, scale_factor, dependencies))
  }
};

#undef SYCL_FFT_DISPATCH

/**
 * A descriptor containing FFT problem parameters.
 *
 * @tparam Scalar type of the scalar used for computations
 * @tparam Domain domain of the FFT
 */
template <typename Scalar, domain Domain>
struct descriptor {
  std::vector<std::size_t> lengths;
  Scalar forward_scale = 1;
  Scalar backward_scale = 1;
  std::size_t number_of_transforms = 1;
  complex_storage complex_storage = complex_storage::COMPLEX;
  placement placement = placement::OUT_OF_PLACE;
  std::vector<std::size_t> forward_strides;
  std::vector<std::size_t> backward_strides;
  std::size_t forward_distance = 1;
  std::size_t backward_distance = 1;
  // TODO: add TRANSPOSE, WORKSPACE and ORDERING if we determine they make sense

  /**
   * Construct a new descriptor object.
   *
   * @param lengths size of the FFT transform
   */
  explicit descriptor(std::vector<std::size_t> lengths) : lengths(lengths), forward_strides{1}, backward_strides{1} {
    // TODO: properly set default values for forward_strides, backward_strides, forward_distance, backward_distance
    forward_distance = lengths[0];
    backward_distance = lengths[0];
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

}  // namespace sycl_fft

#endif
