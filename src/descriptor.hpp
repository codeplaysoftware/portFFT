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
#include <common/subgroup.hpp>
#include <enums.hpp>
#include <sycl/sycl.hpp>
#include <utils.hpp>

#include <complex>
#include <cstdint>
#include <functional>
#include <mutex>
#include <numeric>
#include <vector>

namespace portfft {

namespace detail {

// kernel names
template <typename Scalar, domain Domain, direction Dir, detail::memory, detail::transpose TransposeIn,
          detail::transpose TransposeOut, bool ApplyLoadCallback, bool ApplyStoreCallback, int SubgroupSize,
          int kernel_id = 0>
class workitem_kernel;
template <typename Scalar, domain Domain, direction Dir, detail::memory, detail::transpose TransposeIn,
          detail::transpose TransposeOut, bool ApplyLoadCallback, bool ApplyStoreCallback, int SubgroupSize,
          int kernel_id = 0>
class subgroup_kernel;
template <typename Scalar, domain Domain, direction Dir, detail::memory, detail::transpose TransposeIn,
          detail::transpose TransposeOut, bool ApplyLoadCallback, bool ApplyStoreCallback, int SubgroupSize,
          int kernel_id = 0>
class workgroup_kernel;
template <typename Scalar, domain Domain, direction Dir, detail::memory, detail::transpose TransposeIn,
          detail::transpose TransposeOut, bool ApplyLoadCallback, bool ApplyStoreCallback, int SubgroupSize,
          int kernel_id = 0>
class device_kernel;

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
  std::size_t n_compute_units;
  std::vector<std::size_t> supported_sg_sizes;
  int used_sg_size;
  std::shared_ptr<Scalar> twiddles_forward;
  std::shared_ptr<Scalar> scratch;
  detail::level level;
  std::vector<int> factors;
  std::vector<detail::level> levels;
  std::vector<std::size_t> local_mem_per_factor;
  std::vector<std::pair<sycl::range<1>, sycl::range<1>>> launch_configurations;
  sycl::kernel_bundle<sycl::bundle_state::executable> exec_bundle;
  std::size_t num_sgs_per_wg;
  std::size_t local_memory_size;
  std::size_t l2_cache_size;
  std::size_t num_batches_in_l2;

  template <typename Impl, typename... Args>
  auto dispatch(Args&&... args) {
    switch (level) {
      case detail::level::WORKITEM:
        return Impl::template inner<detail::level::WORKITEM, void>::execute(*this, args...);
      case detail::level::SUBGROUP:
        return Impl::template inner<detail::level::SUBGROUP, void>::execute(*this, args...);
      case detail::level::WORKGROUP:
        return Impl::template inner<detail::level::WORKGROUP, void>::execute(*this, args...);
      case detail::level::DEVICE:
        return Impl::template inner<detail::level::DEVICE, void>::execute(*this, args...);
    }
  }

  template <typename Impl, detail::transpose TransposeIn, typename... Args>
  auto dispatch(Args&&... args) {
    switch (level) {
      case detail::level::WORKITEM:
        return Impl::template inner<detail::level::WORKITEM, TransposeIn, void>::execute(*this, args...);
      case detail::level::SUBGROUP:
        return Impl::template inner<detail::level::SUBGROUP, TransposeIn, void>::execute(*this, args...);
      case detail::level::WORKGROUP:
        return Impl::template inner<detail::level::WORKGROUP, TransposeIn, void>::execute(*this, args...);
      case detail::level::DEVICE:
        return Impl::template inner<detail::level::DEVICE, TransposeIn, void>::execute(*this, args...);
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
  template <int SubgroupSize>
  detail::level prepare_implementation(std::vector<sycl::kernel_id>& ids) {
    factors.clear();
    std::size_t fft_size = params.lengths[0];
    if (!detail::cooley_tukey_size_list_t::has_size(fft_size)) {
      throw std::runtime_error("FFT size " + std::to_string(fft_size) + " is not supported!");
    }
    if (detail::fits_in_wi<Scalar>(fft_size)) {
      detail::get_ids<detail::workitem_kernel, Scalar, Domain, SubgroupSize>(ids);
      return detail::level::WORKITEM;
    }
    int factor_sg = detail::factorize_sg(static_cast<int>(fft_size), SubgroupSize);
    int factor_wi = static_cast<int>(fft_size) / factor_sg;
    if (detail::fits_in_sg<Scalar>(fft_size, SubgroupSize)) {
      // This factorization is duplicated in the dispatch logic on the device.
      // The CT and spec constant factors should match.
      factors.push_back(factor_wi);
      factors.push_back(factor_sg);
      detail::get_ids<detail::subgroup_kernel, Scalar, Domain, SubgroupSize>(ids);
      return detail::level::SUBGROUP;
    }
    std::size_t N = detail::factorize(fft_size);
    std::size_t M = fft_size / N;
    int factor_sg_N = detail::factorize_sg(static_cast<int>(N), SubgroupSize);
    int factor_wi_N = static_cast<int>(N) / factor_sg_N;
    int factor_sg_M = detail::factorize_sg(static_cast<int>(M), SubgroupSize);
    int factor_wi_M = static_cast<int>(M) / factor_sg_M;
    std::size_t total_local_mem_usage = 2 * (fft_size + N + M) * sizeof(Scalar);
    if (detail::fits_in_wi<Scalar>(factor_wi_N) && detail::fits_in_wi<Scalar>(factor_wi_M) &&
        (total_local_mem_usage <= local_memory_size)) {
      factors.push_back(factor_wi_N);
      factors.push_back(factor_sg_N);
      factors.push_back(factor_wi_M);
      factors.push_back(factor_sg_M);
      // This factorization of N and M is duplicated in the dispatch logic on the device.
      // The CT and spec constant factors should match.
      detail::get_ids<detail::workgroup_kernel, Scalar, Domain, SubgroupSize>(ids);
      return detail::level::WORKGROUP;
    } else {
      // Global impl is designed to completely rely on already optimized lower level
      // Implementations. Majority of the performance gains for global implementation will come from the way
      // the input is factored, currently its heuristically set to prefer subgroup impl. mainly because of lower
      // number of sub batches generated
      auto fits_in_target_level = [this](std::size_t size, bool transposed_in = true) -> bool {
        if (detail::fits_in_wi<Scalar>(size))
          return true;
        else
          return detail::fits_in_sg<Scalar>(size, SubgroupSize) && [=, this]() {
            if (transposed_in)
              return local_memory_size >=
                     num_scalars_in_local_mem_struct::template inner<
                         detail::level::SUBGROUP, detail::transpose::TRANSPOSED, void>::execute(*this);
            else
              return local_memory_size >=
                     num_scalars_in_local_mem_struct::template inner<
                         detail::level::SUBGROUP, detail::transpose::NOT_TRANSPOSED, void>::execute(*this);
          }() && !PORTFFT_SLOW_SG_SHUFFLES;
      };

      auto select_impl = [&]<int kernel_id>(std::size_t input_size) -> void {
        if (detail::fits_in_wi<Scalar>(input_size)) {
          levels.push_back(detail::level::WORKITEM);
          detail::get_ids<detail::workitem_kernel, Scalar, Domain, SubgroupSize, kernel_id>(ids);
          return;
        }
        if (detail::fits_in_sg<Scalar>(input_size, SubgroupSize)) {
          levels.push_back(detail::level::SUBGROUP);
          detail::get_ids<detail::subgroup_kernel, Scalar, Domain, SubgroupSize, kernel_id>(ids);
          return;
        }
      };

      detail::factorize_input_struct<0, decltype(fits_in_target_level), decltype(select_impl)>::execute(
          params.lengths[0], fits_in_target_level, select_impl);
      std::size_t num_twiddles = 0;
      for (auto iter = factors.begin(); iter + 1 != factors.end(); iter++) {
        num_twiddles += *iter * std::accumulate(iter + 1, factors.end(), 1, std::multiplies<std::size_t>()) * 2;
      }
      num_batches_in_l2 =
          std::min(static_cast<std::size_t>(PORTFFT_MAX_CONCURRENT_KERNELS),
                   std::max(static_cast<std::size_t>(1), (l2_cache_size - num_twiddles * sizeof(Scalar)) /
                                                             (2 * sizeof(Scalar) * params.lengths[0])));
      return detail::level::DEVICE;
    }
  }

  /**
   * Struct for dispatching `set_spec_constants()` call.
   */
  struct set_spec_constants_struct{
     // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template<detail::level lev, typename Dummy>
    struct inner{
      static void execute(committed_descriptor& desc,
                                    sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle);
    };
  };
  
  /**
   * Sets the implementation dependant specialization constant values.
   *
   * @param in_bundle kernel bundle to set the specialization constants on
   */
  void set_spec_constants(sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle){
    dispatch<set_spec_constants_struct>(in_bundle);
  }

  /**
   * Struct for dispatching `num_scalars_in_local_mem()` call.
   */
  struct num_scalars_in_local_mem_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template <detail::level lev, detail::transpose TransposeIn, typename Dummy, typename... params>
    struct inner {
      static std::size_t execute(committed_descriptor& desc, params...);
    };
  };

  struct num_scalars_in_local_mem_impl_struct {
    template <detail::level Level, detail::transpose TransposeIn, typename Dummy>
    struct inner {
      static std::size_t execute(committed_descriptor& desc, std::size_t fft_size);
    };
  };

  /**
   * Determine the number of scalars we need to have space for in the local memory. It may also modify `num_sgs_in_wg`
   * to make the problem fit in the local memory.
   *
   * @return std::size_t the number of scalars
   */
  template <detail::transpose TransposeIn>
  std::size_t num_scalars_in_local_mem() {
    return dispatch<num_scalars_in_local_mem_struct, TransposeIn>();
  }

  /**
   * Struct for dispatching `calculate_twiddles()` call.
   */
  struct calculate_twiddles_struct{
     // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template<detail::level lev, typename Dummy>
    struct inner{
      static Scalar* execute(committed_descriptor& desc); 
    };
  };

  /**
   * Calculates twiddle factors for the implementation in use.
   *
   * @return Scalar* USM pointer to the twiddle factors
   */
  Scalar* calculate_twiddles() {
    return dispatch<calculate_twiddles_struct>();
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
      used_sg_size = SubgroupSize;
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
        local_memory_size(dev.get_info<sycl::info::device::local_mem_size>()),
        // compile the kernels
        exec_bundle(build_w_spec_const<PORTFFT_SUBGROUP_SIZES>()),
        num_sgs_per_wg(PORTFFT_SGS_IN_WG) {
    // TODO: check and support all the parameter values
    if (params.lengths.size() != 1) {
      throw std::runtime_error("portFFT only supports 1D FFT for now");
    }

    // get some properties we will use for tuning
    n_compute_units = dev.get_info<sycl::info::device::max_compute_units>();
    l2_cache_size = queue.get_device().get_info<sycl::info::device::global_mem_cache_size>();
    twiddles_forward = std::shared_ptr<Scalar>(calculate_twiddles(), [queue](Scalar* ptr) {
      if (ptr != nullptr) {
        sycl::free(ptr, queue);
      }
    });
    scratch = std::shared_ptr<Scalar>(
        sycl::malloc_device<Scalar>(params.lengths[0] * params.number_of_transforms, queue), [queue](Scalar* ptr) {
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
  static constexpr domain domain_value = Domain;

  /**
   * Destructor
   */
  ~committed_descriptor() { queue.wait(); }

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
    throw std::runtime_error("SYCL_FFT: Real to complex FFTs not yet implemented.");
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
    throw std::runtime_error("SYCL_FFT: Real to complex FFTs not yet implemented.");
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
   * @tparam T_in Type of the input buffer or USM pointer
   * @tparam T_out Type of the output buffer or USM pointer
   * @param in buffer or USM pointer to memory containing input data
   * @param out buffer or USM pointer to memory containing output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event
   */
  template <direction Dir, typename T_in, typename T_out>
  sycl::event dispatch_kernel(const T_in in, T_out out, const std::vector<sycl::event>& dependencies = {}) {
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
        return run_kernel<Dir, detail::transpose::NOT_TRANSPOSED, detail::transpose::NOT_TRANSPOSED, SubgroupSize>(
            in, out, scale_factor, dependencies);
      } else if (input_distance == 1 && output_distance == fft_size && in != out) {
        return run_kernel<Dir, detail::transpose::TRANSPOSED, detail::transpose::NOT_TRANSPOSED, SubgroupSize>(
            in, out, scale_factor, dependencies);
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
   * Struct for dispatching `run_kernel()` call.
   * 
   * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam TransposeIn whether input is transposed (interpreting it as a matrix of batch size times FFT size)
   * @tparam SubgroupSize size of the subgroup
   * @tparam T_in Type of the input USM pointer or buffer
   * @tparam T_out Type of the output USM pointer or buffer
   */
  template <direction Dir, detail::transpose TransposeIn, detail::transpose TransposeOut, int SubgroupSize,
            bool ApplyLoadCallback, bool ApplyStoreCallback, typename T_in, typename T_out>
  struct run_kernel_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template<detail::level lev, typename Dummy>
    struct inner{
      static sycl::event execute(committed_descriptor& desc, const T_in& in, T_out& out, Scalar scale_factor,
                                    const std::vector<sycl::event>& dependencies);
    };
  };

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
  template <direction Dir, detail::transpose TransposeIn,
            detail::transpose TransposeOut = detail::transpose::NOT_TRANSPOSED, int SubgroupSize,
            bool ApplyLoadCallback = false, bool ApplyStoreCallback = true, typename T_in, typename T_out>
  sycl::event run_kernel(const T_in& in, T_out& out, Scalar scale_factor,
                         const std::vector<sycl::event>& dependencies) {
    return dispatch<run_kernel_struct<Dir, TransposeIn, TransposeOut, SubgroupSize, ApplyLoadCallback,
                                      ApplyStoreCallback, T_in, T_out>>(in, out, scale_factor, dependencies);
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

}  // namespace portfft

#endif
