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

#include <kernels.hpp>
#include <specialization_constants.hpp>

#include <common/cooley_tukey_compiled_sizes.hpp>
#include <common/exceptions.hpp>
#include <common/subgroup.hpp>
#include <enums.hpp>
#include <utils.hpp>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

namespace portfft {

template <typename, domain>
class committed_descriptor;

namespace detail {
template <typename Scalar, direction, domain Domain, detail::transpose, detail::transpose, apply_load_modifier,
          apply_store_modifier, apply_scale_factor, int, typename TIn>
void dispatch_compute_kernels(committed_descriptor<Scalar, Domain>&, const TIn&, Scalar, std::size_t, std::size_t,
                              std::size_t, std::size_t, std::vector<sycl::event>&);

template <typename Scalar, domain Domain, int SubgroupSize, typename TOut>
void dispatch_transpose_kernels(committed_descriptor<Scalar, Domain>&, TOut&, std::size_t, std::size_t,
                                std::vector<sycl::event>&);

}  // namespace detail

// forward declaration
template <typename, domain>
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

  template <typename ScalarType, direction, domain DomainType, detail::transpose, detail::transpose,
            detail::apply_load_modifier, detail::apply_store_modifier, detail::apply_scale_factor, int, typename TIn>
  friend void detail::dispatch_compute_kernels(committed_descriptor<ScalarType, DomainType>&, const TIn&, ScalarType,
                                               std::size_t, std::size_t, std::size_t, std::size_t,
                                               std::vector<sycl::event>&);

  template <typename ScalarType, domain DomainType, int SubgroupSize, typename TOut>
  friend void detail::dispatch_transpose_kernels(committed_descriptor<ScalarType, DomainType>&, TOut&, std::size_t,
                                                 std::size_t, std::vector<sycl::event>&);

  descriptor<Scalar, Domain> params;
  sycl::queue queue;
  sycl::device dev;
  sycl::context ctx;
  std::size_t local_memory_size;
  std::size_t n_compute_units;
  std::vector<std::size_t> supported_sg_sizes;
  int used_sg_size;
  std::shared_ptr<Scalar> twiddles_forward;
  std::shared_ptr<Scalar> scratch_1;
  std::shared_ptr<Scalar> scratch_2;
  std::shared_ptr<std::size_t> dev_factors;
  detail::level level;
  std::vector<std::size_t> factors;
  std::vector<std::size_t> sub_batches;
  std::vector<std::size_t> inclusive_scan;
  std::vector<detail::level> levels;
  std::vector<std::size_t> local_mem_per_factor;
  std::vector<std::pair<sycl::range<1>, sycl::range<1>>> launch_configurations;
  std::vector<sycl::kernel_bundle<sycl::bundle_state::executable>> exec_bundle;
  std::vector<sycl::kernel_bundle<sycl::bundle_state::executable>> transpose_kernel_bundle;
  std::size_t num_sgs_per_wg;
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
      case detail::level::GLOBAL:
        return Impl::template inner<detail::level::GLOBAL, void>::execute(*this, args...);
      default:
        throw std::runtime_error("Unimplemented!");
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
      case detail::level::GLOBAL:
        return Impl::template inner<detail::level::GLOBAL, TransposeIn, void>::execute(*this, args...);
      default:
        throw std::runtime_error("Unimplemented!");
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
  detail::level prepare_implementation(std::vector<std::vector<sycl::kernel_id>>& ids) {
    factors.clear();

    // TODO: check and support all the parameter values
    if constexpr (Domain != domain::COMPLEX) {
      throw unsupported_configuration("portFFT only supports complex to complex transforms");
    }
    if (params.lengths.size() != 1) {
      throw unsupported_configuration("portFFT only supports 1D FFT for now");
    }
    std::size_t fft_size = params.lengths[0];
    if (!detail::cooley_tukey_size_list_t::has_size(fft_size)) {
      throw unsupported_configuration("FFT size " + std::to_string(fft_size) + " is not supported!");
    }

    if (detail::fits_in_wi<Scalar>(fft_size)) {
      ids.push_back(detail::get_ids<detail::workitem_kernel, Scalar, Domain, SubgroupSize>());
      return detail::level::WORKITEM;
    }
    int factor_sg = detail::factorize_sg(static_cast<int>(fft_size), SubgroupSize);
    int factor_wi = static_cast<int>(fft_size) / factor_sg;
    if (detail::fits_in_sg<Scalar>(fft_size, SubgroupSize)) {
      // This factorization is duplicated in the dispatch logic on the GLOBAL.
      // The CT and spec constant factors should match.
      factors.push_back(static_cast<std::size_t>(factor_wi));
      factors.push_back(static_cast<std::size_t>(factor_sg));
      ids.push_back(detail::get_ids<detail::subgroup_kernel, Scalar, Domain, SubgroupSize>());
      return detail::level::SUBGROUP;
    }
    std::size_t n = detail::factorize(fft_size);
    std::size_t m = fft_size / n;
    int factor_sg_n = detail::factorize_sg(static_cast<int>(n), SubgroupSize);
    int factor_wi_n = static_cast<int>(n) / factor_sg_n;
    int factor_sg_m = detail::factorize_sg(static_cast<int>(m), SubgroupSize);
    int factor_wi_m = static_cast<int>(m) / factor_sg_m;
    if (detail::fits_in_wi<Scalar>(factor_wi_n) && detail::fits_in_wi<Scalar>(factor_wi_m) &&
        (2 * (fft_size + m + n) * sizeof(Scalar) < local_memory_size)) {
      factors.push_back(static_cast<std::size_t>(factor_wi_n));
      factors.push_back(static_cast<std::size_t>(factor_sg_n));
      factors.push_back(static_cast<std::size_t>(factor_wi_m));
      factors.push_back(static_cast<std::size_t>(factor_sg_m));
      // This factorization of N and M is duplicated in the dispatch logic on the device.
      // The CT and spec constant factors should match.
      ids.push_back(detail::get_ids<detail::workgroup_kernel, Scalar, Domain, SubgroupSize>());
      return detail::level::WORKGROUP;
    }
    auto fits_in_target_level = [this](std::size_t size, bool transposed_in = true) -> bool {
      if (detail::fits_in_wi<Scalar>(size)) {
        return true;
      };
      return detail::fits_in_sg<Scalar>(size, SubgroupSize) && [=, this]() {
        if (transposed_in) {
          return local_memory_size >=
                 (2 * num_scalars_in_local_mem_struct::template inner<
                          detail::level::SUBGROUP, detail::transpose::TRANSPOSED, void>::execute(*this)) *
                     sizeof(Scalar);
        }
        return local_memory_size >=
               num_scalars_in_local_mem_struct::template inner<
                   detail::level::SUBGROUP, detail::transpose::NOT_TRANSPOSED, void>::execute(*this) *
                   sizeof(Scalar);
      }() && !PORTFFT_SLOW_SG_SHUFFLES;
    };

    auto select_impl = [&](std::size_t input_size) -> void {
      if (detail::fits_in_wi<Scalar>(input_size)) {
        levels.push_back(detail::level::WORKITEM);
        ids.push_back(detail::get_ids<detail::global_kernel, Scalar, Domain, SubgroupSize>());
        factors.push_back(input_size);
        return;
      }
      if (detail::fits_in_sg<Scalar>(input_size, SubgroupSize)) {
        levels.push_back(detail::level::SUBGROUP);
        ids.push_back(detail::get_ids<detail::global_kernel, Scalar, Domain, SubgroupSize>());
        factors.push_back(input_size);
        return;
      }
    };
    detail::factorize_input_struct<decltype(fits_in_target_level), decltype(select_impl)>::execute(
        params.lengths[0], fits_in_target_level, select_impl);
    std::size_t num_twiddles = 0;
    for (std::size_t i = 0; i < factors.size() - 1; i++) {
      auto batches_at_level = std::accumulate(factors.begin() + static_cast<long>(i) + 1, factors.end(), std::size_t(1),
                                              std::multiplies<std::size_t>());
      sub_batches.push_back(batches_at_level);
      num_twiddles += factors[i] * batches_at_level * 2;
    }
    sub_batches.push_back(factors[factors.size() - 2]);
    num_batches_in_l2 = std::min(static_cast<std::size_t>(PORTFFT_MAX_CONCURRENT_KERNELS),
                                 std::max(static_cast<std::size_t>(1), (l2_cache_size - num_twiddles * sizeof(Scalar)) /
                                                                           (2 * sizeof(Scalar) * params.lengths[0])));
    inclusive_scan.push_back(factors[0]);
    for (std::size_t i = 1; i < factors.size(); i++) {
      inclusive_scan.push_back(inclusive_scan.at(i - 1) * factors.at(i));
    }
    return detail::level::GLOBAL;
  }

  /**
   * Struct for dispatching `set_spec_constants()` call.
   */
  struct set_spec_constants_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template <detail::level Lev, typename Dummy>
    struct inner {
      static void execute(committed_descriptor& desc,
                          std::vector<sycl::kernel_bundle<sycl::bundle_state::input>>& in_bundle);
    };
  };

  /**
   * Sets the implementation dependant specialization constant values.
   *
   * @param in_bundle kernel bundle to set the specialization constants on
   */
  void set_spec_constants(std::vector<sycl::kernel_bundle<sycl::bundle_state::input>>& in_bundle) {
    dispatch<set_spec_constants_struct>(in_bundle);
  }

  /**
   * Struct for dispatching `num_scalars_in_local_mem()` call.
   */
  struct num_scalars_in_local_mem_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template <detail::level Lev, detail::transpose TransposeIn, typename Dummy, typename... Params>
    struct inner {
      static std::size_t execute(committed_descriptor& desc, Params...);
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

  template <int SubgroupSize, int... OtherSGSizes>
  sycl::kernel_bundle<sycl::bundle_state::executable> build_w_spec_const_impl(
      sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle) {
    if (std::count(supported_sg_sizes.begin(), supported_sg_sizes.end(), SubgroupSize)) {
      used_sg_size = SubgroupSize;
      try {
        return sycl::build(in_bundle);
      } catch (std::exception& e) {
        std::cerr << "Build for subgroup size " << SubgroupSize << " failed with message:\n" << e.what() << std::endl;
      }
    }
    if constexpr (sizeof...(OtherSGSizes) == 0) {
      throw std::runtime_error("None of the compiled subgroup sizes are supported by the device!");
    } else {
      return build_w_spec_const_impl<OtherSGSizes...>(in_bundle);
    }
  }

  /**
   * Builds the kernel bundle with appropriate values of specialization constants for the first supported subgroup size.
   *
   * @tparam SubgroupSize first subgroup size
   * @tparam OtherSGSizes other subgroup sizes
   * @return sycl::kernel_bundle<sycl::bundle_state::executable>
   */
  template <int SubgroupSize, int... OtherSGSizes>
  void build_w_spec_const() {
    // This function is called from constructor initializer list and it accesses other data members of the class. These
    // are already initialized by the time this is called only if they are declared in the class definition before the
    // member that is initialized by this function.
    std::vector<std::vector<sycl::kernel_id>> ids;
    std::vector<sycl::kernel_bundle<sycl::bundle_state::input>> input_bundles;
    level = prepare_implementation<SubgroupSize>(ids);
    for (const auto& kernel_ids : ids) {
      if (sycl::is_compatible(kernel_ids, dev)) {
        input_bundles.push_back(sycl::get_kernel_bundle<sycl::bundle_state::input>(queue.get_context(), kernel_ids));
      }
    }
    set_spec_constants(input_bundles);
    for (auto in_bundle : input_bundles) {
      exec_bundle.push_back(build_w_spec_const_impl<SubgroupSize, OtherSGSizes...>(in_bundle));
    }

    if (level == detail::level::GLOBAL) {
      std::vector<sycl::kernel_id> transpose_kernel_ids;
      for (std::size_t i = 0; i < factors.size(); i++) {
        transpose_kernel_ids.clear();
        detail::get_transpose_kernel_ids<Scalar, Domain, SubgroupSize>(transpose_kernel_ids);
        transpose_kernel_bundle.push_back(
            build_transpose_kernel<SubgroupSize, OtherSGSizes...>(transpose_kernel_ids, i));
      }
    }
  }

  template <int SubgroupSize, int... OtherSGSizes>
  sycl::kernel_bundle<sycl::bundle_state::executable> build_transpose_kernel(
      std::vector<sycl::kernel_id>& transpose_kernel_id, std::size_t level_num) {
    auto transpose_in_bundle =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(queue.get_context(), transpose_kernel_id);
    transpose_in_bundle.template set_specialization_constant<detail::GlobalSpecConstLevelNum>(level_num);
    transpose_in_bundle.template set_specialization_constant<detail::GlobalSpecConstNumFactors>(factors.size());
    return build_w_spec_const_impl<SubgroupSize, OtherSGSizes...>(transpose_in_bundle);
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
        local_memory_size(dev.get_info<sycl::info::device::local_mem_size>()),
        // get some properties we will use for tunning
        n_compute_units(dev.get_info<sycl::info::device::max_compute_units>()),
        supported_sg_sizes(dev.get_info<sycl::info::device::sub_group_sizes>()),
        // compile the kernels
        num_sgs_per_wg(PORTFFT_SGS_IN_WG) {
    // TODO: check and support all the parameter values
    if (params.lengths.size() != 1) {
      throw std::runtime_error("portFFT only supports 1D FFT for now");
    }
    build_w_spec_const<PORTFFT_SUBGROUP_SIZES>();
    // get some properties we will use for tuning
    n_compute_units = dev.get_info<sycl::info::device::max_compute_units>();
    std::size_t minimum_local_mem_required;
    if (params.forward_distance == 1 || params.backward_distance == 1) {
      if (2 * params.lengths[0] * sizeof(Scalar) > local_memory_size) {
        throw std::runtime_error("Strided support not available for large sized FFTs");
      }
      minimum_local_mem_required = num_scalars_in_local_mem<detail::transpose::TRANSPOSED>() * sizeof(Scalar);
    } else {
      minimum_local_mem_required = num_scalars_in_local_mem<detail::transpose::NOT_TRANSPOSED>() * sizeof(Scalar);
    }
    if (minimum_local_mem_required > local_memory_size) {
      if (params.forward_distance == 1 || params.backward_distance == 1) {
        throw std::runtime_error("Insufficient amount of local memory available: " + std::to_string(local_memory_size) +
                                 "B. Required: " + std::to_string(minimum_local_mem_required) + "B.");
      }
    }
    if (level == detail::level::GLOBAL) {
      if (queue.is_in_order()) {
        std::cerr << "Out of order queue is required for large FFTs to ensure maximum parallelism " << std::endl;
      }
    }
    twiddles_forward = std::shared_ptr<Scalar>(calculate_twiddles(), [queue](Scalar* ptr) {
      if (ptr != nullptr) {
        sycl::free(ptr, queue);
      }
    });
    if (level == detail::level::GLOBAL) {
      scratch_1 = std::shared_ptr<Scalar>(
          sycl::malloc_device<Scalar>(2 * params.lengths[0] * params.number_of_transforms, queue),
          [queue](Scalar* ptr) {
            if (ptr != nullptr) {
              sycl::free(ptr, queue);
            }
          });
      scratch_2 = std::shared_ptr<Scalar>(
          sycl::malloc_device<Scalar>(2 * params.lengths[0] * params.number_of_transforms, queue),
          [queue](Scalar* ptr) {
            if (ptr != nullptr) {
              sycl::free(ptr, queue);
            }
          });
      dev_factors = std::shared_ptr<std::size_t>(sycl::malloc_device<std::size_t>(3 * factors.size(), queue),
                                                 [queue](std::size_t* ptr) {
                                                   if (ptr != nullptr) {
                                                     sycl::free(ptr, queue);
                                                   }
                                                 });
      queue.copy(factors.data(), dev_factors.get(), factors.size());
      queue.copy(sub_batches.data(), dev_factors.get() + factors.size(), sub_batches.size());
      queue.copy(inclusive_scan.data(), dev_factors.get() + 2 * factors.size(), inclusive_scan.size());
      queue.wait();
    }
  }

  /**
   * Helper function for copy assignment and copy constructor
   * @param desc committed_descriptor to be copied
   */
  void create_copy(const committed_descriptor<Scalar, Domain>& desc) {
#define COPY(x) x = desc.x;
    COPY(params)
    COPY(queue)
    COPY(dev)
    COPY(ctx)
    COPY(local_memory_size)
    COPY(n_compute_units)
    COPY(supported_sg_sizes)
    COPY(used_sg_size)
    COPY(twiddles_forward)
    COPY(dev_factors)
    COPY(level)
    COPY(factors)
    COPY(sub_batches)
    COPY(inclusive_scan)
    COPY(levels)
    COPY(launch_configurations)
    COPY(exec_bundle)
    COPY(transpose_kernel_bundle)
    COPY(num_sgs_per_wg)
    COPY(l2_cache_size)
    COPY(num_batches_in_l2)

#undef COPY

    if (level == detail::level::GLOBAL) {
      scratch_1 = std::shared_ptr<Scalar>(
          sycl::malloc_device<Scalar>(2 * params.lengths[0] * params.number_of_transforms, queue),
          [captured_queue = this->queue](Scalar* ptr) {
            if (ptr != nullptr) {
              sycl::free(ptr, captured_queue);
            }
          });
      scratch_2 = std::shared_ptr<Scalar>(
          sycl::malloc_device<Scalar>(2 * params.lengths[0] * params.number_of_transforms, queue),
          [captured_queue = this->queue](Scalar* ptr) {
            if (ptr != nullptr) {
              sycl::free(ptr, captured_queue);
            }
          });
    }
  }

 public:
  committed_descriptor<Scalar, Domain>(const committed_descriptor<Scalar, Domain>& desc) : params(desc.params) {
    create_copy(desc);
  }
  committed_descriptor<Scalar, Domain>& operator=(const committed_descriptor<Scalar, Domain>& desc) {
    if (this != &desc) {
      create_copy(desc);
    }
    return *this;
  }
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
  template <direction Dir, typename TIn, typename TOut, int SubgroupSize, int... OtherSGSizes>
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
        return run_kernel<Dir, detail::transpose::NOT_TRANSPOSED, SubgroupSize>(in, out, scale_factor, dependencies);
      }
      if (input_distance == 1 && output_distance == fft_size && in != out) {
        return run_kernel<Dir, detail::transpose::TRANSPOSED, SubgroupSize>(in, out, scale_factor, dependencies);
      }
      throw unsupported_configuration("Only contiguous or transposed transforms are supported");
    }
    if constexpr (sizeof...(OtherSGSizes) == 0) {
      throw std::runtime_error("None of the compiled subgroup sizes are supported by the device!");
    } else {
      return dispatch_kernel_helper<Dir, TIn, TOut, OtherSGSizes...>(in, out, dependencies);
    }
  }

  /**
   * Struct for dispatching `run_kernel()` call.
   *
   * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam TransposeIn whether input is transposed (interpreting it as a matrix of batch size times FFT size)
   * @tparam SubgroupSize size of the subgroup
   * @tparam TIn Type of the input USM pointer or buffer
   * @tparam TOut Type of the output USM pointer or buffer
   */
  template <direction Dir, detail::transpose TransposeIn, int SubgroupSize, typename TIn, typename TOut>
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
   * @tparam TransposeIn whether input is transposed (interpreting it as a matrix of batch size times FFT size)
   * @tparam SubgroupSize size of the subgroup
   * @tparam TIn Type of the input USM pointer or buffer
   * @tparam TOut Type of the output USM pointer or buffer
   * @param in USM pointer to memory containing input data
   * @param out USM pointer to memory containing output data
   * @param scale_factor Value with which the result of the FFT will be multiplied
   * @param dependencies events that must complete before the computation
   * @return sycl::event
   */
  template <direction Dir, detail::transpose TransposeIn, int SubgroupSize, typename TIn, typename TOut>
  sycl::event run_kernel(const TIn& in, TOut& out, Scalar scale_factor, const std::vector<sycl::event>& dependencies) {
    return dispatch<run_kernel_struct<Dir, TransposeIn, SubgroupSize, TIn, TOut>>(in, out, scale_factor, dependencies);
  }
};

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
