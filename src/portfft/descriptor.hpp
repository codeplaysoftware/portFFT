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

#include <sycl/sycl.hpp>

#include <complex>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include "common/exceptions.hpp"
#include "common/subgroup.hpp"
#include "defines.hpp"
#include "enums.hpp"
#include "specialization_constant.hpp"
#include "utils.hpp"

namespace portfft {
template <typename Scalar, domain Domain>
class committed_descriptor;
namespace detail {
template <typename Scalar, domain Domain, direction Dir, detail::layout LayoutIn, detail::layout LayoutOut,
          Idx SubgroupSize, typename TIn>
std::vector<sycl::event> compute_level(
    const typename committed_descriptor<Scalar, Domain>::kernel_data_struct& kd_struct, TIn input, Scalar* output,
    const Scalar* twiddles_ptr, const IdxGlobal* factors_triple, Scalar scale_factor,
    IdxGlobal intermediate_twiddle_offset, IdxGlobal subimpl_twiddle_offset, IdxGlobal input_global_offset,
    IdxGlobal committed_size, Idx num_batches_in_l2, IdxGlobal n_transforms, IdxGlobal batch_start, Idx factor_id,
    Idx total_factors, const std::vector<sycl::event>& dependencies, sycl::queue& queue);

template <typename Scalar, domain Domain, typename TOut>
sycl::event transpose_level(const typename committed_descriptor<Scalar, Domain>::kernel_data_struct& kd_struct,
                            const Scalar* input, TOut output, const IdxGlobal* factors_triple, IdxGlobal committed_size,
                            Idx num_batches_in_l2, IdxGlobal n_transforms, IdxGlobal batch_start, Idx factor_num,
                            Idx total_factors, IdxGlobal output_offset, sycl::queue& queue,
                            std::shared_ptr<Scalar>& ptr1, std::shared_ptr<Scalar>& ptr2,
                            const std::vector<sycl::event>& events);

// kernel names
// TODO: Remove all templates except Scalar, Domain and Memory and SubgroupSize
template <typename Scalar, domain, direction, detail::memory, detail::layout, detail::layout, Idx SubgroupSize>
class workitem_kernel;
template <typename Scalar, domain, direction, detail::memory, detail::layout, detail::layout, Idx SubgroupSize>
class subgroup_kernel;
template <typename Scalar, domain, direction, detail::memory, detail::layout, detail::layout, Idx SubgroupSize>
class workgroup_kernel;
template <typename Scalar, domain, direction, detail::memory, detail::layout, detail::layout, Idx SubgroupSize>
class global_kernel;
template <typename Scalar, detail::memory>
class transpose_kernel;

/**
 * Return the default strides for a given dft size
 *
 * @param lengths the dimensions of the dft
 */
inline std::vector<std::size_t> get_default_strides(const std::vector<std::size_t>& lengths) {
  std::vector<std::size_t> strides(lengths.size());
  std::size_t total_size = 1;
  for (std::size_t i_plus1 = lengths.size(); i_plus1 > 0; i_plus1--) {
    std::size_t i = i_plus1 - 1;
    strides[i] = total_size;
    total_size *= lengths[i];
  }
  return strides;
}

/**
 * Return whether the given descriptor has default strides and distance for a given direction
 *
 * @tparam Descriptor Descriptor type
 * @param desc Descriptor to check
 * @param dir Direction
 */
template <typename Descriptor>
bool has_default_strides_and_distance(const Descriptor& desc, direction dir) {
  const auto default_strides = get_default_strides(desc.lengths);
  const auto default_distance = desc.get_flattened_length();
  return desc.get_strides(dir) == default_strides && desc.get_distance(dir) == default_distance;
}

/**
 * Return whether the given descriptor has strides and distance consistent with the batch interleaved layout
 *
 * @tparam Descriptor Descriptor type
 * @param desc Descriptor to check
 * @param dir Direction
 */
template <typename Descriptor>
bool is_batch_interleaved(const Descriptor& desc, direction dir) {
  return desc.lengths.size() == 1 && desc.get_distance(dir) == 1 &&
         desc.get_strides(dir).back() == desc.number_of_transforms;
}

/**
 * Return an enum describing the layout of the data in the descriptor
 *
 * @tparam Descriptor Descriptor type
 * @param desc Descriptor to check
 * @param dir Direction
 */
template <typename Descriptor>
detail::layout get_layout(const Descriptor& desc, direction dir) {
  if (has_default_strides_and_distance(desc, dir)) {
    return detail::layout::PACKED;
  }
  if (is_batch_interleaved(desc, dir)) {
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
  template <typename Scalar1, domain Domain1, direction Dir, detail::layout LayoutIn, detail::layout LayoutOut,
            Idx SubgroupSize, typename TIn>
  friend std::vector<sycl::event> detail::compute_level(
      const typename committed_descriptor<Scalar1, Domain1>::kernel_data_struct& kd_struct, TIn input, Scalar1* output,
      const Scalar1* twiddles_ptr, const IdxGlobal* factors_triple, Scalar1 scale_factor,
      IdxGlobal intermediate_twiddle_offset, IdxGlobal subimpl_twiddle_offset, IdxGlobal input_global_offset,
      IdxGlobal committed_size, Idx num_batches_in_l2, IdxGlobal n_transforms, IdxGlobal batch_start, Idx factor_id,
      Idx total_factors, const std::vector<sycl::event>& dependencies, sycl::queue& queue);

  template <typename Scalar1, domain Domain1, typename TOut>
  friend sycl::event detail::transpose_level(
      const typename committed_descriptor<Scalar1, Domain1>::kernel_data_struct& kd_struct, const Scalar1* input,
      TOut output, const IdxGlobal* factors_triple, IdxGlobal committed_size, Idx num_batches_in_l2,
      IdxGlobal n_transforms, IdxGlobal batch_start, Idx factor_num, Idx total_factors, IdxGlobal output_offset,
      sycl::queue& queue, std::shared_ptr<Scalar1>& ptr1, std::shared_ptr<Scalar1>& ptr2,
      const std::vector<sycl::event>& events);

  descriptor<Scalar, Domain> params;
  sycl::queue queue;
  sycl::device dev;
  sycl::context ctx;
  Idx n_compute_units;
  std::vector<std::size_t> supported_sg_sizes;
  Idx local_memory_size;
  IdxGlobal llc_size;
  std::shared_ptr<Scalar> scratch_ptr_1;
  std::shared_ptr<Scalar> scratch_ptr_2;
  std::size_t scratch_space_required;

  struct kernel_data_struct {
    sycl::kernel_bundle<sycl::bundle_state::executable> exec_bundle;
    std::vector<Idx> factors;
    std::size_t length;
    Idx used_sg_size;
    Idx num_sgs_per_wg;
    std::shared_ptr<Scalar> twiddles_forward;
    detail::level level;
    IdxGlobal batch_size;
    std::size_t local_mem_required;
    IdxGlobal global_range;
    IdxGlobal local_range;

    kernel_data_struct(sycl::kernel_bundle<sycl::bundle_state::executable>&& exec_bundle,
                       const std::vector<Idx>& factors, std::size_t length, Idx used_sg_size, Idx num_sgs_per_wg,
                       std::shared_ptr<Scalar> twiddles_forward, detail::level level)
        : exec_bundle(std::move(exec_bundle)),
          factors(factors),
          length(length),
          used_sg_size(used_sg_size),
          num_sgs_per_wg(num_sgs_per_wg),
          twiddles_forward(twiddles_forward),
          level(level) {}
  };

  struct dimension_struct {
    std::vector<kernel_data_struct> kernels;
    std::shared_ptr<IdxGlobal> factors_and_scan;
    detail::level level;
    std::size_t length;
    Idx used_sg_size;
    Idx num_batches_in_l2;
    Idx num_factors;

    dimension_struct(std::vector<kernel_data_struct> kernels, detail::level level, std::size_t length, Idx used_sg_size)
        : kernels(kernels), level(level), length(length), used_sg_size(used_sg_size) {}
  };

  std::vector<dimension_struct> dimensions;

  template <typename Impl, typename... Args>
  auto dispatch(detail::level level, Args&&... args) {
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
        // This should be unreachable
        throw unsupported_configuration("Unimplemented");
    }
  }

  template <typename Impl, detail::layout LayoutIn, typename... Args>
  auto dispatch(detail::level level, Args&&... args) {
    switch (level) {
      case detail::level::WORKITEM:
        return Impl::template inner<detail::level::WORKITEM, LayoutIn, void>::execute(*this, args...);
      case detail::level::SUBGROUP:
        return Impl::template inner<detail::level::SUBGROUP, LayoutIn, void>::execute(*this, args...);
      case detail::level::WORKGROUP:
        return Impl::template inner<detail::level::WORKGROUP, LayoutIn, void>::execute(*this, args...);
      case detail::level::GLOBAL:
        return Impl::template inner<detail::level::GLOBAL, LayoutIn, void>::execute(*this, args...);
      default:
        // This should be unreachable
        throw unsupported_configuration("Unimplemented");
    }
  }

  template <typename Impl, detail::layout LayoutIn, detail::layout LayoutOut, Idx SubgroupSize, typename... Args>
  auto dispatch(detail::level level, Args&&... args) {
    switch (level) {
      case detail::level::WORKITEM:
        return Impl::template inner<detail::level::WORKITEM, LayoutIn, LayoutOut, SubgroupSize, void>::execute(*this,
                                                                                                               args...);
      case detail::level::SUBGROUP:
        return Impl::template inner<detail::level::SUBGROUP, LayoutIn, LayoutOut, SubgroupSize, void>::execute(*this,
                                                                                                               args...);
      case detail::level::WORKGROUP:
        return Impl::template inner<detail::level::WORKGROUP, LayoutIn, LayoutOut, SubgroupSize, void>::execute(
            *this, args...);
      case detail::level::GLOBAL:
        return Impl::template inner<detail::level::GLOBAL, LayoutIn, LayoutOut, SubgroupSize, void>::execute(*this,
                                                                                                             args...);
      default:
        // This should be unreachable
        throw unsupported_configuration("Unimplemented");
    }
  }

  /**
   * Prepares the implementation for the particular problem size. That includes factorizing it and getting ids for the
   * set of kernels that need to be JIT compiled.
   *
   * @tparam SubgroupSize size of the subgroup
   * @param kernel_num the consecutive number of the kernel to prepare
   * @return implementation to use for the dimension and a vector of tuples of: implementation to use for a kernel,
   * vector of kernel ids, factors
   */
  template <Idx SubgroupSize>
  std::tuple<detail::level, std::vector<std::tuple<detail::level, std::vector<sycl::kernel_id>, std::vector<Idx>>>>
  prepare_implementation(std::size_t kernel_num) {
    // TODO: check and support all the parameter values
    if constexpr (Domain != domain::COMPLEX) {
      throw unsupported_configuration("portFFT only supports complex to complex transforms");
    }

    std::vector<sycl::kernel_id> ids;
    std::vector<Idx> factors;
    IdxGlobal fft_size = static_cast<IdxGlobal>(params.lengths[kernel_num]);
    if (detail::fits_in_wi<Scalar>(fft_size)) {
      ids = detail::get_ids<detail::workitem_kernel, Scalar, Domain, SubgroupSize>();
      return {detail::level::WORKITEM, {{detail::level::WORKITEM, ids, factors}}};
    }
    if (detail::fits_in_sg<Scalar>(fft_size, SubgroupSize)) {
      Idx factor_sg = detail::factorize_sg(static_cast<Idx>(fft_size), SubgroupSize);
      Idx factor_wi = static_cast<Idx>(fft_size) / factor_sg;
      // This factorization is duplicated in the dispatch logic on the device.
      // The CT and spec constant factors should match.
      factors.push_back(factor_wi);
      factors.push_back(factor_sg);
      ids = detail::get_ids<detail::subgroup_kernel, Scalar, Domain, SubgroupSize>();
      return {detail::level::SUBGROUP, {{detail::level::SUBGROUP, ids, factors}}};
    }
    IdxGlobal n_idx_global = detail::factorize(fft_size);
    if (detail::can_cast_safely<IdxGlobal, Idx>(n_idx_global) &&
        detail::can_cast_safely<IdxGlobal, Idx>(fft_size / n_idx_global)) {
      if (n_idx_global == 1) {
        throw unsupported_configuration("FFT size ", fft_size, " : Large Prime sized FFT currently is unsupported");
      }
      Idx n = static_cast<Idx>(n_idx_global);
      Idx m = static_cast<Idx>(fft_size / n_idx_global);
      Idx factor_sg_n = detail::factorize_sg(n, SubgroupSize);
      Idx factor_wi_n = n / factor_sg_n;
      Idx factor_sg_m = detail::factorize_sg(m, SubgroupSize);
      Idx factor_wi_m = m / factor_sg_m;
      Idx temp_num_sgs_in_wg;
      std::size_t local_memory_usage = num_scalars_in_local_mem<detail::layout::PACKED>(
                                           detail::level::WORKGROUP, static_cast<std::size_t>(fft_size), SubgroupSize,
                                           {factor_sg_n, factor_wi_n, factor_sg_m, factor_wi_m}, temp_num_sgs_in_wg) *
                                       sizeof(Scalar);
      // Checks for PACKED layout only at the moment, as the other layout will not be supported
      // by the global implementation. For such sizes, only PACKED layout will be supported
      if (detail::fits_in_wi<Scalar>(factor_wi_n) && detail::fits_in_wi<Scalar>(factor_wi_m) &&
          (local_memory_usage <= static_cast<std::size_t>(local_memory_size))) {
        factors.push_back(factor_wi_n);
        factors.push_back(factor_sg_n);
        factors.push_back(factor_wi_m);
        factors.push_back(factor_sg_m);
        // This factorization of N and M is duplicated in the dispatch logic on the device.
        // The CT and spec constant factors should match.
        ids = detail::get_ids<detail::workgroup_kernel, Scalar, Domain, SubgroupSize>();
        return {detail::level::WORKGROUP, {{detail::level::WORKGROUP, ids, factors}}};
      }
    }
    std::vector<std::tuple<detail::level, std::vector<sycl::kernel_id>, std::vector<Idx>>> param_vec;
    auto check_and_select_target_level = [&](IdxGlobal factor_size, bool batch_interleaved_layout = true) -> bool {
      if (detail::fits_in_wi<Scalar>(factor_size)) {
        param_vec.emplace_back(detail::level::WORKITEM,
                               detail::get_ids<detail::global_kernel, Scalar, Domain, SubgroupSize>(),
                               std::vector<Idx>{static_cast<Idx>(factor_size)});

        return true;
      }
      bool fits_in_local_memory_subgroup = [&]() {
        Idx temp_num_sgs_in_wg;
        IdxGlobal factor_sg = detail::factorize_sg<IdxGlobal>(factor_size, SubgroupSize);
        IdxGlobal factor_wi = factor_size / factor_sg;
        if (detail::can_cast_safely<IdxGlobal, Idx>(factor_sg) && detail::can_cast_safely<IdxGlobal, Idx>(factor_wi)) {
          if (batch_interleaved_layout) {
            return (2 *
                        num_scalars_in_local_mem<detail::layout::BATCH_INTERLEAVED>(
                            detail::level::SUBGROUP, static_cast<std::size_t>(factor_size), SubgroupSize,
                            {static_cast<Idx>(factor_sg), static_cast<Idx>(factor_wi)}, temp_num_sgs_in_wg) *
                        sizeof(Scalar) +
                    2 * static_cast<std::size_t>(factor_size) * sizeof(Scalar)) <
                   static_cast<std::size_t>(local_memory_size);
          }
          return (num_scalars_in_local_mem<detail::layout::PACKED>(
                      detail::level::SUBGROUP, static_cast<std::size_t>(factor_size), SubgroupSize,
                      {static_cast<Idx>(factor_sg), static_cast<Idx>(factor_wi)}, temp_num_sgs_in_wg) *
                      sizeof(Scalar) +
                  2 * static_cast<std::size_t>(factor_size) * sizeof(Scalar)) <
                 static_cast<std::size_t>(local_memory_size);
        }
        return false;
      }();
      if (detail::fits_in_sg<Scalar>(factor_size, SubgroupSize) && fits_in_local_memory_subgroup &&
          !PORTFFT_SLOW_SG_SHUFFLES) {
        param_vec.emplace_back(detail::level::SUBGROUP,
                               detail::get_ids<detail::global_kernel, Scalar, Domain, SubgroupSize>(),
                               std::vector<Idx>{detail::factorize_sg(static_cast<Idx>(factor_size), SubgroupSize),
                                                static_cast<Idx>(factor_size) /
                                                    detail::factorize_sg(static_cast<Idx>(factor_size), SubgroupSize)});
        return true;
      }
      return false;
    };
    detail::factorize_input(fft_size, check_and_select_target_level);
    return {detail::level::GLOBAL, param_vec};
  }

  /**
   * Struct for dispatching `set_spec_constants()` call.
   */
  struct set_spec_constants_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template <detail::level Lev, typename Dummy>
    struct inner {
      static void execute(committed_descriptor& desc, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle,
                          std::size_t length, const std::vector<Idx>& factors, detail::level level, Idx factor_num,
                          Idx num_factors);
    };
  };

  /**
   * Sets the implementation dependant specialization constant value
   * @param top_level implementation to dispatch to
   * @param in_bundle input kernel bundle to set spec constants for
   * @param length length of the fft
   * @param factors factors of the corresponsing length
   * @param multiply_on_load Whether the input data is multiplied with some data array before fft computation.
   * @param multiply_on_store Whether the input data is multiplied with some data array after fft computation.
   * @param scale_factor_applied whether or not to multiply scale factor
   * @param level sub implementation to run which will be set as a spec constant.
   * @param factor_num factor number which is set as a spec constant
   * @param num_factors total number of factors of the committed size, set as a spec constant.
   */
  void set_spec_constants(detail::level top_level, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle,
                          std::size_t length, const std::vector<Idx>& factors,
                          detail::elementwise_multiply multiply_on_load, detail::elementwise_multiply multiply_on_store,
                          detail::apply_scale_factor scale_factor_applied, detail::level level, Idx factor_num = 0,
                          Idx num_factors = 0) {
    const Idx length_idx = static_cast<Idx>(length);
    // These spec constants are used in all implementations, so we set them here
    in_bundle.template set_specialization_constant<detail::SpecConstComplexStorage>(params.complex_storage);
    in_bundle.template set_specialization_constant<detail::SpecConstNumRealsPerFFT>(2 * length_idx);
    in_bundle.template set_specialization_constant<detail::SpecConstWIScratchSize>(2 * detail::wi_temps(length_idx));
    in_bundle.template set_specialization_constant<detail::SpecConstMultiplyOnLoad>(multiply_on_load);
    in_bundle.template set_specialization_constant<detail::SpecConstMultiplyOnStore>(multiply_on_store);
    in_bundle.template set_specialization_constant<detail::SpecConstApplyScaleFactor>(scale_factor_applied);
    dispatch<set_spec_constants_struct>(top_level, in_bundle, length, factors, level, factor_num, num_factors);
  }

  /**
   * Struct for dispatching `num_scalars_in_local_mem()` call.
   */
  struct num_scalars_in_local_mem_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template <detail::level Lev, detail::layout LayoutIn, typename Dummy>
    struct inner {
      static std::size_t execute(committed_descriptor& desc, std::size_t length, Idx used_sg_size,
                                 const std::vector<Idx>& factors, Idx& num_sgs_per_wg);
    };
  };

  /**
   * Determine the number of scalars we need to have space for in the local memory. It may also modify `num_sgs_per_wg`
   * to make the problem fit in the local memory.
   *
   * @param level the implementation that will be used
   * @param length length of the FFT the kernel will execute
   * @param used_sg_size subgroup size the kernel will use
   * @param factors factorization of the FFT size the kernel will use
   * @param[out] num_sgs_per_wg number of subgroups in a workgroup
   * @return the number of scalars
   */
  template <detail::layout LayoutIn>
  std::size_t num_scalars_in_local_mem(detail::level level, std::size_t length, Idx used_sg_size,
                                       const std::vector<Idx>& factors, Idx& num_sgs_per_wg) {
    return dispatch<num_scalars_in_local_mem_struct, LayoutIn>(level, length, used_sg_size, factors, num_sgs_per_wg);
  }

  /**
   * Struct for dispatching `calculate_twiddles()` call.
   */
  struct calculate_twiddles_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template <detail::level Lev, typename Dummy>
    struct inner {
      static Scalar* execute(committed_descriptor& desc, dimension_struct& dimension_data);
    };
  };

  /**
   * Calculates twiddle factors for the implementation in use.
   *
   * @param dimension_data data about the dimension for which twiddles are needed
   * @return Scalar* USM pointer to the twiddle factors
   */
  Scalar* calculate_twiddles(dimension_struct& dimension_data) {
    return dispatch<calculate_twiddles_struct>(dimension_data.level, dimension_data);
  }

  /**
   * Builds the kernel bundles with appropriate values of specialization constants for the first supported subgroup
   * size.
   *
   * @tparam SubgroupSize first subgroup size
   * @tparam OtherSGSizes other subgroup sizes
   * @param kernel_num the consecutive number of the kernel to build
   * @return `dimension_struct` for the newly built kernels
   */
  template <Idx SubgroupSize, Idx... OtherSGSizes>
  dimension_struct build_w_spec_const(std::size_t kernel_num) {
    if (std::count(supported_sg_sizes.begin(), supported_sg_sizes.end(), SubgroupSize)) {
      auto [top_level, prepared_vec] = prepare_implementation<SubgroupSize>(kernel_num);
      bool is_compatible = true;
      for (auto [level, ids, factors] : prepared_vec) {
        is_compatible = is_compatible && sycl::is_compatible(ids, dev);
        if (!is_compatible) {
          break;
        }
      }
      std::vector<kernel_data_struct> result;
      if (is_compatible) {
        std::size_t counter = 0;
        for (auto [level, ids, factors] : prepared_vec) {
          auto in_bundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(queue.get_context(), ids);
          if (top_level == detail::level::GLOBAL) {
            if (counter == prepared_vec.size() - 1) {
              set_spec_constants(detail::level::GLOBAL, in_bundle,
                                 static_cast<std::size_t>(
                                     std::accumulate(factors.begin(), factors.end(), Idx(1), std::multiplies<Idx>())),
                                 factors, detail::elementwise_multiply::NOT_APPLIED,
                                 detail::elementwise_multiply::NOT_APPLIED, detail::apply_scale_factor::APPLIED, level,
                                 static_cast<Idx>(counter), static_cast<Idx>(prepared_vec.size()));
            } else {
              set_spec_constants(detail::level::GLOBAL, in_bundle,
                                 static_cast<std::size_t>(
                                     std::accumulate(factors.begin(), factors.end(), Idx(1), std::multiplies<Idx>())),
                                 factors, detail::elementwise_multiply::NOT_APPLIED,
                                 detail::elementwise_multiply::APPLIED, detail::apply_scale_factor::NOT_APPLIED, level,
                                 static_cast<Idx>(counter), static_cast<Idx>(prepared_vec.size()));
            }
          } else {
            set_spec_constants(level, in_bundle, params.lengths[kernel_num], factors,
                               detail::elementwise_multiply::NOT_APPLIED, detail::elementwise_multiply::NOT_APPLIED,
                               detail::apply_scale_factor::APPLIED, level);
          }
          try {
            result.emplace_back(sycl::build(in_bundle), factors, params.lengths[kernel_num], SubgroupSize,
                                PORTFFT_SGS_IN_WG, std::shared_ptr<Scalar>(), level);
          } catch (std::exception& e) {
            std::cerr << "Build for subgroup size " << SubgroupSize << " failed with message:\n"
                      << e.what() << std::endl;
            is_compatible = false;
            break;
          }
          counter++;
        }
        if (is_compatible) {
          return {result, top_level, params.lengths[kernel_num], SubgroupSize};
        }
      }
    }
    if constexpr (sizeof...(OtherSGSizes) == 0) {
      throw invalid_configuration("None of the compiled subgroup sizes are supported by the device");
    } else {
      return build_w_spec_const<OtherSGSizes...>(kernel_num);
    }
  }

  /**
   * Function which calculates the amount of scratch space required, and also pre computes the necessary scans required.
   * @param num_global_level_dimensions number of global level dimensions in the committed size
   */
  void allocate_scratch_and_precompute_scan(Idx num_global_level_dimensions) {
    std::size_t n_kernels = params.lengths.size();
    if (num_global_level_dimensions == 1) {
      std::size_t global_dimension = 0;
      for (std::size_t i = 0; i < n_kernels; i++) {
        if (dimensions.at(i).level == detail::level::GLOBAL) {
          global_dimension = i;
          break;
        }
      }
      std::vector<IdxGlobal> factors;
      std::vector<IdxGlobal> sub_batches;
      std::vector<IdxGlobal> inclusive_scan;
      std::size_t cache_required_for_twiddles = 0;
      for (const auto& kernel_data : dimensions.at(global_dimension).kernels) {
        IdxGlobal factor_size = static_cast<IdxGlobal>(
            std::accumulate(kernel_data.factors.begin(), kernel_data.factors.end(), 1, std::multiplies<Idx>()));
        cache_required_for_twiddles +=
            static_cast<std::size_t>(2 * factor_size * kernel_data.batch_size) * sizeof(Scalar);
        factors.push_back(factor_size);
        sub_batches.push_back(kernel_data.batch_size);
      }
      dimensions.at(global_dimension).num_factors = static_cast<Idx>(factors.size());
      std::size_t cache_space_left_for_batches = static_cast<std::size_t>(llc_size) - cache_required_for_twiddles;
      // TODO: In case of mutli-dim (single dim global sized), this should be batches corresposding to that dim
      dimensions.at(global_dimension).num_batches_in_l2 = static_cast<Idx>(std::min(
          static_cast<std::size_t>(PORTFFT_MAX_CONCURRENT_KERNELS),
          std::min(params.number_of_transforms,
                   std::max(std::size_t(1), cache_space_left_for_batches /
                                                (2 * dimensions.at(global_dimension).length * sizeof(Scalar))))));
      scratch_space_required = 2 * dimensions.at(global_dimension).length *
                               static_cast<std::size_t>(dimensions.at(global_dimension).num_batches_in_l2);
      scratch_ptr_1 =
          detail::make_shared<Scalar>(2 * dimensions.at(global_dimension).length *
                                          static_cast<std::size_t>(dimensions.at(global_dimension).num_batches_in_l2),
                                      queue);
      scratch_ptr_2 =
          detail::make_shared<Scalar>(2 * dimensions.at(global_dimension).length *
                                          static_cast<std::size_t>(dimensions.at(global_dimension).num_batches_in_l2),
                                      queue);
      inclusive_scan.push_back(factors.at(0));
      for (std::size_t i = 1; i < factors.size(); i++) {
        inclusive_scan.push_back(inclusive_scan.at(i - 1) * factors.at(i));
      }
      dimensions.at(global_dimension).factors_and_scan =
          detail::make_shared<IdxGlobal>(factors.size() + sub_batches.size() + inclusive_scan.size(), queue);
      queue.copy(factors.data(), dimensions.at(global_dimension).factors_and_scan.get(), factors.size());
      queue.copy(sub_batches.data(), dimensions.at(global_dimension).factors_and_scan.get() + factors.size(),
                 sub_batches.size());
      queue.copy(inclusive_scan.data(),
                 dimensions.at(global_dimension).factors_and_scan.get() + factors.size() + sub_batches.size(),
                 inclusive_scan.size());
      queue.wait();
      // build transpose kernels
      std::size_t num_transposes_required = factors.size() - 1;
      for (std::size_t i = 0; i < num_transposes_required; i++) {
        std::vector<sycl::kernel_id> ids;
        auto in_bundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(queue.get_context(),
                                                                            detail::get_transpose_kernel_ids<Scalar>());
        in_bundle.template set_specialization_constant<detail::GlobalSpecConstLevelNum>(static_cast<Idx>(i));
        in_bundle.template set_specialization_constant<detail::GlobalSpecConstNumFactors>(
            static_cast<Idx>(factors.size()));
        dimensions.at(global_dimension)
            .kernels.emplace_back(
                sycl::build(in_bundle),
                std::vector<Idx>{static_cast<Idx>(factors.at(i)), static_cast<Idx>(sub_batches.at(i))}, 1, 1, 1,
                std::shared_ptr<Scalar>(), detail::level::GLOBAL);
      }
    } else {
      std::size_t max_encountered_global_size = 0;
      for (std::size_t i = 0; i < n_kernels; i++) {
        if (dimensions.at(i).level == detail::level::GLOBAL) {
          max_encountered_global_size = max_encountered_global_size > dimensions.at(i).length
                                            ? max_encountered_global_size
                                            : dimensions.at(i).length;
        }
      }
      // TODO: max_scratch_size should be max(global_size_1 * corresponding_batches_in_l2, global_size_1 *
      // corresponding_batches_in_l2), in the case of multi-dim global FFTs.
      scratch_space_required = 2 * max_encountered_global_size * params.number_of_transforms;
      scratch_ptr_1 = detail::make_shared<Scalar>(scratch_space_required, queue);
      scratch_ptr_2 = detail::make_shared<Scalar>(scratch_space_required, queue);
      for (std::size_t i = 0; i < n_kernels; i++) {
        if (dimensions.at(i).level == detail::level::GLOBAL) {
          std::vector<IdxGlobal> factors;
          std::vector<IdxGlobal> sub_batches;
          std::vector<IdxGlobal> inclusive_scan;
          for (const auto& kernel_data : dimensions.at(i).kernels) {
            IdxGlobal factor_size = static_cast<IdxGlobal>(
                std::accumulate(kernel_data.factors.begin(), kernel_data.factors.end(), 1, std::multiplies<Idx>()));
            factors.push_back(factor_size);
            sub_batches.push_back(kernel_data.batch_size);
          }
          inclusive_scan.push_back(factors.at(0));
          for (std::size_t j = 1; j < factors.size(); j++) {
            inclusive_scan.push_back(inclusive_scan.at(j - 1) * factors.at(j));
          }
          dimensions.at(i).num_factors = static_cast<Idx>(factors.size());
          dimensions.at(i).factors_and_scan =
              detail::make_shared<IdxGlobal>(factors.size() + sub_batches.size() + inclusive_scan.size(), queue);
          queue.copy(factors.data(), dimensions.at(i).factors_and_scan.get(), factors.size());
          queue.copy(sub_batches.data(), dimensions.at(i).factors_and_scan.get() + factors.size(), sub_batches.size());
          queue.copy(inclusive_scan.data(),
                     dimensions.at(i).factors_and_scan.get() + factors.size() + sub_batches.size(),
                     inclusive_scan.size());
          queue.wait();
          // build transpose kernels
          std::size_t num_transposes_required = factors.size() - 1;
          for (std::size_t j = 0; j < num_transposes_required; j++) {
            auto in_bundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
                queue.get_context(), detail::get_transpose_kernel_ids<Scalar>());
            in_bundle.template set_specialization_constant<detail::GlobalSpecConstLevelNum>(static_cast<Idx>(i));
            in_bundle.template set_specialization_constant<detail::GlobalSpecConstNumFactors>(
                static_cast<Idx>(factors.size()));
            dimensions.at(i).kernels.emplace_back(
                sycl::build(in_bundle),
                std::vector<Idx>{static_cast<Idx>(factors.at(j)), static_cast<Idx>(sub_batches.at(j))}, 1, 1, 1,
                std::shared_ptr<Scalar>(), detail::level::GLOBAL);
          }
        }
      }
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
        local_memory_size(static_cast<Idx>(queue.get_device().get_info<sycl::info::device::local_mem_size>())),
        llc_size(static_cast<IdxGlobal>(queue.get_device().get_info<sycl::info::device::global_mem_cache_size>())) {
    // check it's suitable to run

    const auto forward_layout = detail::get_layout(params, direction::FORWARD);
    const auto backward_layout = detail::get_layout(params, direction::BACKWARD);
    if (params.lengths.size() > 1) {
      const bool supported_layout =
          forward_layout == detail::layout::PACKED && backward_layout == detail::layout::PACKED;
      if (!supported_layout) {
        throw unsupported_configuration("Multi-dimensional transforms are only supported with default data layout");
      }
    } else {
      const bool supported_layout =
          (forward_layout == detail::layout::PACKED || forward_layout == detail::layout::BATCH_INTERLEAVED) &&
          (backward_layout == detail::layout::PACKED || backward_layout == detail::layout::BATCH_INTERLEAVED);
      if (!supported_layout) {
        throw unsupported_configuration("Arbitary strides are not supported");
      }
    }

    // compile the kernels and precalculate twiddles
    std::size_t n_kernels = params.lengths.size();
    for (std::size_t i = 0; i < n_kernels; i++) {
      dimensions.push_back(build_w_spec_const<PORTFFT_SUBGROUP_SIZES>(i));
      dimensions.back().kernels.at(0).twiddles_forward =
          std::shared_ptr<Scalar>(calculate_twiddles(dimensions.back()), [queue](Scalar* ptr) {
            if (ptr != nullptr) {
              sycl::free(ptr, queue);
            }
          });
    }

    bool is_scratch_required = false;
    Idx num_global_level_dimensions = 0;
    for (std::size_t i = 0; i < n_kernels; i++) {
      if (dimensions.at(i).level == detail::level::GLOBAL) {
        is_scratch_required = true;
        num_global_level_dimensions++;
      }
    }
    if (num_global_level_dimensions != 0) {
      if (params.lengths.size() > 1) {
        throw unsupported_configuration("Only 1D FFTs that do not fit in local memory are supported");
      }
      if (params.get_distance(direction::FORWARD) != params.lengths[0] ||
          params.get_distance(direction::BACKWARD) != params.lengths[0]) {
        throw unsupported_configuration("Large FFTs are currently only supported in non-strided format");
      }
    }

    if (is_scratch_required) {
      allocate_scratch_and_precompute_scan(num_global_level_dimensions);
    }
  }

  /**
   * Utility function fo copy constructor and copy assignment operator
   * @param desc committed_descriptor of which the copy is to be made
   */
  void create_copy(const committed_descriptor<Scalar, Domain>& desc) {
#define PORTFFT_COPY(x) this->x = desc.x;
    PORTFFT_COPY(params)
    PORTFFT_COPY(queue)
    PORTFFT_COPY(dev)
    PORTFFT_COPY(ctx)
    PORTFFT_COPY(n_compute_units)
    PORTFFT_COPY(supported_sg_sizes)
    PORTFFT_COPY(local_memory_size)
    PORTFFT_COPY(dimensions)
    PORTFFT_COPY(scratch_space_required)
    PORTFFT_COPY(llc_size)

#undef PORTFFT_COPY
    bool is_scratch_required = false;
    for (std::size_t i = 0; i < desc.dimensions.size(); i++) {
      if (desc.dimensions.at(i).level == detail::level::GLOBAL) {
        is_scratch_required = true;
        break;
      }
    }
    if (is_scratch_required) {
      this->scratch_ptr_1 =
          detail::make_shared<Scalar>(static_cast<std::size_t>(desc.scratch_space_required), this->queue);
      this->scratch_ptr_2 =
          detail::make_shared<Scalar>(static_cast<std::size_t>(desc.scratch_space_required), this->queue);
    }
  }

 public:
  committed_descriptor(const committed_descriptor& desc) : params(desc.params) { create_copy(desc); }
  committed_descriptor& operator=(const committed_descriptor& desc) {
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
   * Computes in-place forward FFT, working on buffers.
   *
   * @param inout_real buffer containing real part of the input and output data
   * @param inout_imag buffer containing imaginary part of the input and output data
   */
  void compute_forward(sycl::buffer<scalar_type, 1>& inout_real, sycl::buffer<scalar_type, 1>& inout_imag) {
    // For now we can just call out-of-place implementation.
    // This might need to be changed once we implement support for large sizes that work in global memory.
    compute_forward(inout_real, inout_imag, inout_real, inout_imag);
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
   * Computes in-place backward FFT, working on buffers.
   *
   * @param inout_real buffer containing real part of the input and output data
   * @param inout_imag buffer containing imaginary part of the input and output data
   */
  void compute_backward(sycl::buffer<scalar_type, 1>& inout_real, sycl::buffer<scalar_type, 1>& inout_imag) {
    // For now we can just call out-of-place implementation.
    // This might need to be changed once we implement support for large sizes that work in global memory.
    compute_backward(inout_real, inout_imag, inout_real, inout_imag);
  }

  /**
   * Computes out-of-place forward FFT, working on buffers.
   *
   * @param in buffer containing input data
   * @param out buffer containing output data
   */
  void compute_forward(const sycl::buffer<complex_type, 1>& in, sycl::buffer<complex_type, 1>& out) {
    dispatch_direction<direction::FORWARD>(in, out, in, out, complex_storage::INTERLEAVED_COMPLEX);
  }

  /**
   * Computes out-of-place forward FFT, working on buffers.
   *
   * @param in_real buffer containing real part of the input data
   * @param in_imag buffer containing imaginary part of the input data
   * @param out_real buffer containing real part of the output data
   * @param out_imag buffer containing imaginary part of the output data
   */
  void compute_forward(const sycl::buffer<scalar_type, 1>& in_real, const sycl::buffer<scalar_type, 1>& in_imag,
                       sycl::buffer<scalar_type, 1>& out_real, sycl::buffer<scalar_type, 1>& out_imag) {
    dispatch_direction<direction::FORWARD>(in_real, out_real, in_imag, out_imag, complex_storage::SPLIT_COMPLEX);
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
    dispatch_direction<direction::BACKWARD>(in, out, in, out, complex_storage::INTERLEAVED_COMPLEX);
  }

  /**
   * Compute out of place backward FFT, working on buffers
   *
   * @param in_real buffer containing real part of the input data
   * @param in_imag buffer containing imaginary part of the input data
   * @param out_real buffer containing real part of the output data
   * @param out_imag buffer containing imaginary part of the output data
   */
  void compute_backward(const sycl::buffer<scalar_type, 1>& in_real, const sycl::buffer<scalar_type, 1>& in_imag,
                        sycl::buffer<scalar_type, 1>& out_real, sycl::buffer<scalar_type, 1>& out_imag) {
    dispatch_direction<direction::BACKWARD>(in_real, out_real, in_imag, out_imag, complex_storage::SPLIT_COMPLEX);
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
   * @param inout_real USM pointer to memory containing real part of the input and output data
   * @param inout_imag USM pointer to memory containing imaginary part of the input and output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_forward(scalar_type* inout_real, scalar_type* inout_imag,
                              const std::vector<sycl::event>& dependencies = {}) {
    // For now we can just call out-of-place implementation.
    // This might need to be changed once we implement support for large sizes that work in global memory.
    return compute_forward(inout_real, inout_imag, inout_real, inout_imag, dependencies);
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
   * Computes in-place backward FFT, working on USM memory.
   *
   * @param inout_real USM pointer to memory containing real part of the input and output data
   * @param inout_imag USM pointer to memory containing imaginary part of the input and output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_backward(scalar_type* inout_real, scalar_type* inout_imag,
                               const std::vector<sycl::event>& dependencies = {}) {
    return compute_backward(inout_real, inout_real, inout_imag, inout_imag, dependencies);
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
    return dispatch_direction<direction::FORWARD>(in, out, in, out, complex_storage::INTERLEAVED_COMPLEX, dependencies);
  }

  /**
   * Computes out-of-place forward FFT, working on USM memory.
   *
   * @param in_real USM pointer to memory containing real part of the input data
   * @param in_imag USM pointer to memory containing imaginary part of the input data
   * @param out_real USM pointer to memory containing real part of the output data
   * @param out_imag USM pointer to memory containing imaginary part of the output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_forward(const scalar_type* in_real, const scalar_type* in_imag, scalar_type* out_real,
                              scalar_type* out_imag, const std::vector<sycl::event>& dependencies = {}) {
    return dispatch_direction<direction::FORWARD>(in_real, out_real, in_imag, out_imag, complex_storage::SPLIT_COMPLEX,
                                                  dependencies);
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
    return dispatch_direction<direction::BACKWARD>(in, out, in, out, complex_storage::INTERLEAVED_COMPLEX,
                                                   dependencies);
  }

  /**
   * Computes out-of-place backward FFT, working on USM memory.
   *
   * @param in_real USM pointer to memory containing real part of the input data
   * @param in_imag USM pointer to memory containing imaginary part of the input data
   * @param out_real USM pointer to memory containing real part of the output data
   * @param out_imag USM pointer to memory containing imaginary part of the output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_backward(const scalar_type* in_real, const scalar_type* in_imag, scalar_type* out_real,
                               scalar_type* out_imag, const std::vector<sycl::event>& dependencies = {}) {
    return dispatch_direction<direction::BACKWARD>(in_real, out_real, in_imag, out_imag, complex_storage::SPLIT_COMPLEX,
                                                   dependencies);
  }

 private:
  /**
   * Dispatches to the implementation for the appropriate direction.
   *
   * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam TIn Type of the input buffer or USM pointer
   * @tparam TOut Type of the output buffer or USM pointer
   * @param in buffer or USM pointer to memory containing input data. Real part of input data if
   * `descriptor.complex_storage` is split.
   * @param out buffer or USM pointer to memory containing output data. Real part of input data if
   * `descriptor.complex_storage` is split.
   * @param in_imag buffer or USM pointer to memory containing imaginary part of the input data. Ignored if
   * `descriptor.complex_storage` is interleaved.
   * @param out_imag buffer or USM pointer to memory containing imaginary part of the output data. Ignored if
   * `descriptor.complex_storage` is interleaved.
   * @param used_storage how components of a complex value are stored - either split or interleaved
   * @param dependencies events that must complete before the computation
   * @return sycl::event
   */
  template <direction Dir, typename TIn, typename TOut>
  sycl::event dispatch_direction(const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                                 complex_storage used_storage, const std::vector<sycl::event>& dependencies = {}) {
    if (used_storage != params.complex_storage) {
      if (used_storage == complex_storage::SPLIT_COMPLEX) {
        throw invalid_configuration(
            "To use interface with split real and imaginary memory, descriptor.complex_storage must be set to "
            "SPLIT_COMPLEX.");
      }
      throw invalid_configuration(
          "To use interface with interleaved real and imaginary values, descriptor.complex_storage must be set to "
          "INTERLEAVED_COMPLEX.");
    }
    if constexpr (Dir == direction::FORWARD) {
      return dispatch_dimensions<Dir>(in, out, in_imag, out_imag, dependencies, params.forward_strides,
                                      params.backward_strides, params.forward_distance, params.backward_distance,
                                      params.forward_offset, params.backward_offset, params.forward_scale);
    } else {
      return dispatch_dimensions<Dir>(in, out, in_imag, out_imag, dependencies, params.backward_strides,
                                      params.forward_strides, params.backward_distance, params.forward_distance,
                                      params.backward_offset, params.forward_offset, params.backward_scale);
    }
  }

  /**
   * Dispatches to the implementation for the appropriate number of dimensions.
   *
   * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam TIn Type of the input buffer or USM pointer
   * @tparam TOut Type of the output buffer or USM pointer
   * @param in buffer or USM pointer to memory containing input data. Real part of input data if
   * `descriptor.complex_storage` is split.
   * @param out buffer or USM pointer to memory containing output data. Real part of input data if
   * `descriptor.complex_storage` is split.
   * @param in_imag buffer or USM pointer to memory containing imaginary part of the input data. Ignored if
   * `descriptor.complex_storage` is interleaved.
   * @param out_imag buffer or USM pointer to memory containing imaginary part of the output data. Ignored if
   * `descriptor.complex_storage` is interleaved.
   * @param dependencies events that must complete before the computation
   * @param input_strides strides between input elements for each dimension of one FFT
   * @param output_strides strides between output elements for each dimension of one FFT
   * @param input_distance distance between the starts of input data for two consecutive FFTs
   * @param output_distance distance between the starts of output data for two consecutive FFTs
   * @param input_offset offset into input allocation where the data for FFTs start
   * @param output_offset offset into output allocation where the data for FFTs start
   * @param scale_factor scaling factor applied to the result
   * @param dimension_data data for the dimension this call will work on
   * @return sycl::event
   */
  template <direction Dir, typename TIn, typename TOut>
  sycl::event dispatch_dimensions(const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                                  const std::vector<sycl::event>& dependencies,
                                  const std::vector<std::size_t>& input_strides,
                                  const std::vector<std::size_t>& output_strides, std::size_t input_distance,
                                  std::size_t output_distance, std::size_t input_offset, std::size_t output_offset,
                                  Scalar scale_factor) {
    using TOutConst = std::conditional_t<std::is_pointer_v<TOut>, const std::remove_pointer_t<TOut>*, const TOut>;
    std::size_t n_dimensions = params.lengths.size();
    std::size_t total_size = params.get_flattened_length();

    const auto forward_layout = detail::get_layout(params, direction::FORWARD);
    const auto backward_layout = detail::get_layout(params, direction::BACKWARD);

    // currently multi-dimensional transforms are implemented just for default (PACKED) data layout
    const bool multi_dim_supported =
        forward_layout == detail::layout::PACKED && backward_layout == detail::layout::PACKED;
    if (n_dimensions != 1 && !multi_dim_supported) {
      throw internal_error("Only default layout is supported for multi-dimensional transforms.");
    }

    // product of sizes of all dimension inner relative to the one we are currently working on
    std::size_t inner_size = 1;
    // product of sizes of all dimension outer relative to the one we are currently working on
    std::size_t outer_size = total_size / params.lengths.back();
    std::size_t input_stride_0 = input_strides.back();
    std::size_t output_stride_0 = output_strides.back();
    // distances are currently used just in the first dimension - these changes are meant for that one
    // TODO fix this to support non-default layouts
    if (input_stride_0 < input_distance) {  // for example: batch interleaved input
      input_distance = params.lengths.back();
    }
    if (output_stride_0 < output_distance) {  // for example: batch interleaved output
      output_distance = params.lengths.back();
    }

    sycl::event previous_event = dispatch_kernel_1d<Dir>(
        in, out, in_imag, out_imag, dependencies, params.number_of_transforms * outer_size, input_stride_0,
        output_stride_0, input_distance, output_distance, input_offset, output_offset, scale_factor, dimensions.back());
    if (n_dimensions == 1) {
      return previous_event;
    }
    std::vector<sycl::event> previous_events{previous_event};
    std::vector<sycl::event> next_events;
    inner_size *= params.lengths.back();
    for (std::size_t i = n_dimensions - 2; i != static_cast<std::size_t>(-1); i--) {
      outer_size /= params.lengths[i];
      // TODO do everything from the next loop in a single kernel once we support more than one distance in the
      // kernels.
      std::size_t stride_between_kernels = inner_size * params.lengths[i];
      for (std::size_t j = 0; j < params.number_of_transforms * outer_size; j++) {
        sycl::event e = dispatch_kernel_1d<Dir, TOutConst, TOut>(
            out, out, out_imag, out_imag, previous_events, inner_size, inner_size, inner_size, 1, 1,
            output_offset + j * stride_between_kernels, output_offset + j * stride_between_kernels,
            static_cast<Scalar>(1.0), dimensions[i]);
        next_events.push_back(e);
      }
      inner_size *= params.lengths[i];
      std::swap(previous_events, next_events);
      next_events.clear();
    }
    return queue.single_task(previous_events, []() {});  // just to get an event that depends on all previous ones
  }

  /**
   * Dispatches the kernel with the first subgroup size that is supported by the device.
   *
   * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam TIn Type of the input buffer or USM pointer
   * @tparam TOut Type of the output buffer or USM pointer
   * @param in buffer or USM pointer to memory containing input data. Real part of input data if
   * `descriptor.complex_storage` is split.
   * @param out buffer or USM pointer to memory containing output data. Real part of input data if
   * `descriptor.complex_storage` is split.
   * @param in_imag buffer or USM pointer to memory containing imaginary part of the input data. Ignored if
   * `descriptor.complex_storage` is interleaved.
   * @param out_imag buffer or USM pointer to memory containing imaginary part of the output data. Ignored if
   * `descriptor.complex_storage` is interleaved.
   * @param dependencies events that must complete before the computation
   * @param n_transforms number of FT transforms to do in one call
   * @param input_stride stride between input elements of one FFT
   * @param output_stride stride between output elements of one FFT
   * @param input_distance distance between the starts of input data for two consecutive FFTs
   * @param output_distance distance between the starts of output data for two consecutive FFTs
   * @param input_offset offset into input allocation where the data for FFTs start
   * @param output_offset offset into output allocation where the data for FFTs start
   * @param scale_factor scaling factor applied to the result
   * @param dimension_data data for the dimension this call will work on
   * @return sycl::event
   */
  template <direction Dir, typename TIn, typename TOut>
  sycl::event dispatch_kernel_1d(const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                                 const std::vector<sycl::event>& dependencies, std::size_t n_transforms,
                                 std::size_t input_stride, std::size_t output_stride, std::size_t input_distance,
                                 std::size_t output_distance, std::size_t input_offset, std::size_t output_offset,
                                 Scalar scale_factor, dimension_struct& dimension_data) {
    return dispatch_kernel_1d_helper<Dir, TIn, TOut, PORTFFT_SUBGROUP_SIZES>(
        in, out, in_imag, out_imag, dependencies, n_transforms, input_stride, output_stride, input_distance,
        output_distance, input_offset, output_offset, scale_factor, dimension_data);
  }

  /**
   * Helper for dispatching the kernel with the first subgroup size that is supported by the device.
   *
   * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam TIn Type of the input buffer or USM pointer
   * @tparam TOut Type of the output buffer or USM pointer
   * @tparam SubgroupSize first subgroup size
   * @tparam OtherSGSizes other subgroup sizes
   * @param in buffer or USM pointer to memory containing input data. Real part of input data if
   * `descriptor.complex_storage` is split.
   * @param out buffer or USM pointer to memory containing output data. Real part of input data if
   * `descriptor.complex_storage` is split.
   * @param in_imag buffer or USM pointer to memory containing imaginary part of the input data. Ignored if
   * `descriptor.complex_storage` is interleaved.
   * @param out_imag buffer or USM pointer to memory containing imaginary part of the output data. Ignored if
   * `descriptor.complex_storage` is interleaved.
   * @param dependencies events that must complete before the computation
   * @param n_transforms number of FT transforms to do in one call
   * @param input_stride stride between input elements of one FFT
   * @param output_stride stride between output elements of one FFT
   * @param input_distance distance between the starts of input data for two consecutive FFTs
   * @param output_distance distance between the starts of output data for two consecutive FFTs
   * @param input_offset offset into input allocation where the data for FFTs start
   * @param output_offset offset into output allocation where the data for FFTs start
   * @param scale_factor scaling factor applied to the result
   * @param dimension_data data for the dimension this call will work on
   * @return sycl::event
   */
  template <direction Dir, typename TIn, typename TOut, Idx SubgroupSize, Idx... OtherSGSizes>
  sycl::event dispatch_kernel_1d_helper(const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                                        const std::vector<sycl::event>& dependencies, std::size_t n_transforms,
                                        std::size_t input_stride, std::size_t output_stride, std::size_t input_distance,
                                        std::size_t output_distance, std::size_t input_offset,
                                        std::size_t output_offset, Scalar scale_factor,
                                        dimension_struct& dimension_data) {
    if (SubgroupSize == dimension_data.used_sg_size) {
      const bool input_packed = input_distance == dimension_data.length && input_stride == 1;
      const bool output_packed = output_distance == dimension_data.length && output_stride == 1;
      const bool input_batch_interleaved = input_distance == 1 && input_stride == n_transforms;
      const bool output_batch_interleaved = output_distance == 1 && output_stride == n_transforms;
      for (kernel_data_struct kernel_data : dimension_data.kernels) {
        std::size_t minimum_local_mem_required;
        if (input_batch_interleaved) {
          minimum_local_mem_required = num_scalars_in_local_mem<detail::layout::BATCH_INTERLEAVED>(
                                           kernel_data.level, kernel_data.length, SubgroupSize, kernel_data.factors,
                                           kernel_data.num_sgs_per_wg) *
                                       sizeof(Scalar);
          if (static_cast<Idx>(minimum_local_mem_required) > local_memory_size) {
            throw out_of_local_memory_error(
                "Insufficient amount of local memory available: " + std::to_string(local_memory_size) +
                "B. Required: " + std::to_string(minimum_local_mem_required) + "B.");
          }
        }
      }
      if (input_packed && output_packed) {
        return run_kernel<Dir, detail::layout::PACKED, detail::layout::PACKED, SubgroupSize>(
            in, out, in_imag, out_imag, dependencies, n_transforms, input_offset, output_offset, scale_factor,
            dimension_data);
      }
      if (input_batch_interleaved && output_packed && in != out) {
        return run_kernel<Dir, detail::layout::BATCH_INTERLEAVED, detail::layout::PACKED, SubgroupSize>(
            in, out, in_imag, out_imag, dependencies, n_transforms, input_offset, output_offset, scale_factor,
            dimension_data);
      }
      if (input_packed && output_batch_interleaved && in != out) {
        return run_kernel<Dir, detail::layout::PACKED, detail::layout::BATCH_INTERLEAVED, SubgroupSize>(
            in, out, in_imag, out_imag, dependencies, n_transforms, input_offset, output_offset, scale_factor,
            dimension_data);
      }
      if (input_batch_interleaved && output_batch_interleaved) {
        return run_kernel<Dir, detail::layout::BATCH_INTERLEAVED, detail::layout::BATCH_INTERLEAVED, SubgroupSize>(
            in, out, in_imag, out_imag, dependencies, n_transforms, input_offset, output_offset, scale_factor,
            dimension_data);
      }
      throw unsupported_configuration("Only PACKED or BATCH_INTERLEAVED transforms are supported");
    }
    if constexpr (sizeof...(OtherSGSizes) == 0) {
      throw invalid_configuration("None of the compiled subgroup sizes are supported by the device!");
    } else {
      return dispatch_kernel_1d_helper<Dir, TIn, TOut, OtherSGSizes...>(
          in, out, in_imag, out_imag, dependencies, n_transforms, input_stride, output_stride, input_distance,
          output_distance, input_offset, output_offset, scale_factor, dimension_data);
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
      static sycl::event execute(committed_descriptor& desc, const TIn& in, TOut& out, const TIn& in_imag,
                                 TOut& out_imag, const std::vector<sycl::event>& dependencies, std::size_t n_transforms,
                                 std::size_t forward_offset, std::size_t backward_offset, Scalar scale_factor,
                                 dimension_struct& dimension_data);
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
   * @param in buffer or USM pointer to memory containing input data. Real part of input data if
   * `descriptor.complex_storage` is split.
   * @param out buffer or USM pointer to memory containing output data. Real part of input data if
   * `descriptor.complex_storage` is split.
   * @param in_imag buffer or USM pointer to memory containing imaginary part of the input data. Ignored if
   * `descriptor.complex_storage` is interleaved.
   * @param out_imag buffer or USM pointer to memory containing imaginary part of the output data. Ignored if
   * `descriptor.complex_storage` is interleaved.
   * @param dependencies events that must complete before the computation
   * @param n_transforms number of FT transforms to do in one call
   * @param input_offset offset into input allocation where the data for FFTs start
   * @param output_offset offset into output allocation where the data for FFTs start
   * @param scale_factor scaling factor applied to the result
   * @param dimension_data data for the dimension this call will work on
   * @return sycl::event
   */
  template <direction Dir, detail::layout LayoutIn, detail::layout LayoutOut, Idx SubgroupSize, typename TIn,
            typename TOut>
  sycl::event run_kernel(const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                         const std::vector<sycl::event>& dependencies, std::size_t n_transforms,
                         std::size_t input_offset, std::size_t output_offset, Scalar scale_factor,
                         dimension_struct& dimension_data) {
    // mixing const and non-const inputs leads to hard-to-debug linking errors, as both use the same kernel name, but
    // are called from different template instantiations.
    static_assert(!std::is_pointer_v<TIn> || std::is_const_v<std::remove_pointer_t<TIn>>,
                  "We do not differentiate kernel names between kernels with const and non-const USM inputs, so all "
                  "should be const");
    // kernel names currently assume both are the same. Mixing them without adding TOut to kernel names would lead to
    // hard-to-debug linking errors
    static_assert(std::is_pointer_v<TIn> == std::is_pointer_v<TOut>,
                  "Both input and output to the kernels should be the same - either buffers or USM");
    using TInReinterpret = decltype(detail::reinterpret<const Scalar>(in));
    using TOutReinterpret = decltype(detail::reinterpret<Scalar>(out));
    std::size_t vec_multiplier = params.complex_storage == complex_storage::INTERLEAVED_COMPLEX ? 2 : 1;
    return dispatch<run_kernel_struct<Dir, LayoutIn, LayoutOut, SubgroupSize, TInReinterpret, TOutReinterpret>>(
        dimension_data.level, detail::reinterpret<const Scalar>(in), detail::reinterpret<Scalar>(out),
        detail::reinterpret<const Scalar>(in_imag), detail::reinterpret<Scalar>(out_imag), dependencies,
        static_cast<IdxGlobal>(n_transforms), static_cast<IdxGlobal>(vec_multiplier * input_offset),
        static_cast<IdxGlobal>(vec_multiplier * output_offset), scale_factor, dimension_data);
  }
};

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
  static_assert(std::is_floating_point_v<Scalar>, "Scalar must be a floating point type");

  /**
   * FFT domain.
   * Determines whether the input (resp. output) is real or complex in the forward (resp. backward) direction.
   */
  static constexpr domain Domain = DescDomain;

  /**
   * The lengths in elements of each dimension, ordered from most to least significant (i.e. contiguous dimension last).
   * N-D transforms are supported. Must be specified.
   */
  std::vector<std::size_t> lengths;
  /**
   * A scaling factor applied to the output of forward transforms. Default value is 1.
   */
  Scalar forward_scale = 1;
  /**
   * A scaling factor applied to the output of backward transforms. Default value is 1.
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
   * The data layout of complex values. Default value is complex_storage::INTERLEAVED_COMPLEX.
   * complex_storage::INTERLEAVED_COMPLEX indicates that the real and imaginary part of a complex number is contiguous
   * i.e an Array of Structures. complex_storage::SPLIT_COMPLEX indicates that all the real values are contiguous and
   * all the imaginary values are contiguous i.e. a Structure of Arrays.
   */
  complex_storage complex_storage = complex_storage::INTERLEAVED_COMPLEX;
  /**
   * Indicates if the memory address of the output pointer is the same as the input pointer. Default value is
   * placement::OUT_OF_PLACE. When placement::OUT_OF_PLACE is used, only the out of place compute_xxxward functions can
   * be used and the memory pointed to by the input pointer and the memory pointed to by the output pointer must not
   * overlap at all. When placement::IN_PLACE is used, only the in-place compute_xxxward functions can be used.
   */
  placement placement = placement::OUT_OF_PLACE;
  /**
   * The strides of the data in the forward domain in elements.
   * For offset s0 and distance m, for strides[s1,s2,...,sd] the element in batch b at index [i1,i2,...,id] is located
   * at elems[s0 + m*b + s1*i1 + s2*i2 + ... + sd*id]. The default value for a d-dimensional transform is
   * {prod(lengths[0..d-1]), prod(lengths[0..d-2]), ..., lengths[0]*lengths[1], lengths[0], 1}, where prod is the
   * product. Only the default value is supported for transforms with more than one dimension. Strides do not include
   * the offset.
   */
  std::vector<std::size_t> forward_strides;
  /**
   * The strides of the data in the backward domain in elements.
   * For offset s0 and distance m, for strides[s1,s2,...,sd] the element in batch b at index [i1,i2,...,id] is located
   * at elems[s0 + m*b + s1*i1 + s2*i2 + ... + sd*id]. The default value for a d-dimensional transform is
   * {prod(lengths[0..d-1]), prod(lengths[0..d-2]), ..., lengths[0]*lengths[1], lengths[0], 1}, where prod is the
   * product. Only the default value is supported for transforms with more than one dimension. Strides do not include
   * the offset.
   */
  std::vector<std::size_t> backward_strides;
  /**
   * The number of elements between the first value of each transform in the forward domain. The default value is
   * the product of the lengths. Must be either 1 or the product of the lengths.
   * Only the default value is supported for transforms with more than one dimension.
   * For a d-dimensional transform, exactly one of `forward_strides[d-1]` and `forward_distance` must be 1.
   */
  std::size_t forward_distance = 1;
  /**
   * The number of elements between the first value of each transform in the backward domain. The default value
   * is the product of the lengths. Must be the same as forward_distance.
   * Only the default value is supported for transforms with more than one dimension.
   */
  std::size_t backward_distance = 1;
  /**
   * The number of elements between the start of memory allocation for data in forward domain and the first value
   * to use for FFT computation. The default value is 0.
   */
  std::size_t forward_offset = 0;
  /**
   * The number of elements between the start of memory allocation for data in backward domain and the first value
   * to use for FFT computation. The default value is 0.
   */
  std::size_t backward_offset = 0;
  // TODO: add TRANSPOSE, WORKSPACE and ORDERING if we determine they make sense

  /**
   * Construct a new descriptor object.
   *
   * @param lengths size of the FFT transform
   */
  explicit descriptor(const std::vector<std::size_t>& lengths)
      : lengths(lengths), forward_strides(detail::get_default_strides(lengths)), backward_strides(forward_strides) {
    // TODO: properly set default values for distances for real transforms
    std::size_t total_size = get_flattened_length();
    forward_distance = total_size;
    backward_distance = total_size;
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
    return get_buffer_count(get_strides(dir), get_distance(dir), get_offset(dir));
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
   * Return the offset for a given direction
   *
   * @param dir direction
   */
  std::size_t get_offset(direction dir) const noexcept {
    return dir == direction::FORWARD ? forward_offset : backward_offset;
  }

  /**
   * Return a mutable reference to the offset for a given direction
   *
   * @param dir direction
   */
  std::size_t& get_offset(direction dir) noexcept {
    return dir == direction::FORWARD ? forward_offset : backward_offset;
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
   * Compute the number of elements required for a buffer with the descriptor's length, number of transforms and the
   * given strides and distance.
   * The number of elements is the same irrespective of the FFT domain.
   *
   * @param strides buffer's strides
   * @param distance buffer's distance
   */
  std::size_t get_buffer_count(const std::vector<std::size_t>& strides, std::size_t distance,
                               std::size_t offset) const noexcept {
    // Compute the last element that can be accessed
    std::size_t last_elt_idx = (number_of_transforms - 1) * distance;
    for (std::size_t i = 0; i < lengths.size(); ++i) {
      last_elt_idx += (lengths[i] - 1) * strides[i];
    }
    return offset + last_elt_idx + 1;
  }
};

}  // namespace portfft

#endif
