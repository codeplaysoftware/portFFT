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

#ifndef PORTFFT_COMMITTED_DESCRIPTOR_IMPL_HPP
#define PORTFFT_COMMITTED_DESCRIPTOR_IMPL_HPP

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
struct descriptor;

namespace detail {

template <typename Scalar, domain Domain>
class committed_descriptor_impl;

template <typename Scalar, domain Domain, Idx SubgroupSize, typename TIn>
std::vector<sycl::event> compute_level(
    const typename committed_descriptor_impl<Scalar, Domain>::kernel_data_struct& kd_struct, const TIn& input,
    Scalar* output, const TIn& input_imag, Scalar* output_imag, const Scalar* twiddles_ptr,
    const IdxGlobal* factors_triple, IdxGlobal intermediate_twiddle_offset, IdxGlobal subimpl_twiddle_offset,
    IdxGlobal input_global_offset, IdxGlobal committed_size, Idx num_batches_in_l2, IdxGlobal n_transforms,
    IdxGlobal batch_start, Idx factor_id, Idx total_factors, complex_storage storage,
    const std::vector<sycl::event>& dependencies, sycl::queue& queue);

template <typename Scalar, domain Domain, typename TOut>
sycl::event transpose_level(const typename committed_descriptor_impl<Scalar, Domain>::kernel_data_struct& kd_struct,
                            const Scalar* input, TOut output, const IdxGlobal* factors_triple, IdxGlobal committed_size,
                            Idx num_batches_in_l2, IdxGlobal n_transforms, IdxGlobal batch_start, Idx total_factors,
                            IdxGlobal output_offset, sycl::queue& queue, const std::vector<sycl::event>& events,
                            complex_storage storage);

// kernel names
template <typename Scalar, domain, detail::memory, Idx SubgroupSize>
class workitem_kernel;
template <typename Scalar, domain, detail::memory, Idx SubgroupSize>
class subgroup_kernel;
template <typename Scalar, domain, detail::memory, Idx SubgroupSize>
class workgroup_kernel;
template <typename Scalar, domain, detail::memory, Idx SubgroupSize>
class global_kernel;
template <typename Scalar, detail::memory>
class transpose_kernel;

/**
 * A committed descriptor that contains everything that is needed to run FFT.
 *
 * @tparam Scalar type of the scalar used for computations
 * @tparam Domain domain of the FFT
 */
template <typename Scalar, domain Domain>
class committed_descriptor_impl {
  friend struct descriptor<Scalar, Domain>;
  template <typename Scalar1, domain Domain1, Idx SubgroupSize, typename TIn>
  friend std::vector<sycl::event> detail::compute_level(
      const typename committed_descriptor_impl<Scalar1, Domain1>::kernel_data_struct& kd_struct, const TIn& input,
      Scalar1* output, const TIn& input_imag, Scalar1* output_imag, const Scalar1* twiddles_ptr,
      const IdxGlobal* factors_triple, IdxGlobal intermediate_twiddle_offset, IdxGlobal subimpl_twiddle_offset,
      IdxGlobal input_global_offset, IdxGlobal committed_size, Idx num_batches_in_l2, IdxGlobal n_transforms,
      IdxGlobal batch_start, Idx factor_id, Idx total_factors, complex_storage storage,
      const std::vector<sycl::event>& dependencies, sycl::queue& queue);

  template <typename Scalar1, domain Domain1, typename TOut>
  friend sycl::event detail::transpose_level(
      const typename committed_descriptor_impl<Scalar1, Domain1>::kernel_data_struct& kd_struct, const Scalar1* input,
      TOut output, const IdxGlobal* factors_triple, IdxGlobal committed_size, Idx num_batches_in_l2,
      IdxGlobal n_transforms, IdxGlobal batch_start, Idx total_factors, IdxGlobal output_offset, sycl::queue& queue,
      const std::vector<sycl::event>& events, complex_storage storage);

  /**
   * Vector containing the sub-implementation level, kernel_ids and factors for each factor that requires a separate
   * kernel.
   */
  using kernel_ids_and_metadata_t =
      std::vector<std::tuple<detail::level, std::vector<sycl::kernel_id>, std::vector<Idx>>>;
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
    std::vector<kernel_data_struct> forward_kernels;
    std::vector<kernel_data_struct> backward_kernels;
    std::vector<kernel_data_struct> transpose_kernels;
    std::shared_ptr<IdxGlobal> factors_and_scan;
    detail::level level;
    std::size_t length;
    Idx used_sg_size;
    Idx num_batches_in_l2;
    Idx num_factors;

    dimension_struct(std::vector<kernel_data_struct> forward_kernels, std::vector<kernel_data_struct> backward_kernels,
                     detail::level level, std::size_t length, Idx used_sg_size)
        : forward_kernels(std::move(forward_kernels)),
          backward_kernels(std::move(backward_kernels)),
          level(level),
          length(length),
          used_sg_size(used_sg_size) {}
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

  template <typename Impl, Idx SubgroupSize, typename... Args>
  auto dispatch(detail::level level, Args&&... args) {
    switch (level) {
      case detail::level::WORKITEM:
        return Impl::template inner<detail::level::WORKITEM, SubgroupSize, void>::execute(*this, args...);
      case detail::level::SUBGROUP:
        return Impl::template inner<detail::level::SUBGROUP, SubgroupSize, void>::execute(*this, args...);
      case detail::level::WORKGROUP:
        return Impl::template inner<detail::level::WORKGROUP, SubgroupSize, void>::execute(*this, args...);
      case detail::level::GLOBAL:
        return Impl::template inner<detail::level::GLOBAL, SubgroupSize, void>::execute(*this, args...);
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
  std::tuple<detail::level, kernel_ids_and_metadata_t> prepare_implementation(std::size_t kernel_num) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    // TODO: check and support all the parameter values
    if constexpr (Domain != domain::COMPLEX) {
      throw unsupported_configuration("portFFT only supports complex to complex transforms");
    }

    std::vector<sycl::kernel_id> ids;
    std::vector<Idx> factors;
    IdxGlobal fft_size = static_cast<IdxGlobal>(params.lengths[kernel_num]);
    if (detail::fits_in_wi<Scalar>(fft_size)) {
      ids = detail::get_ids<detail::workitem_kernel, Scalar, Domain, SubgroupSize>();
      PORTFFT_LOG_TRACE("Prepared workitem impl for size: ", fft_size);
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
      PORTFFT_LOG_TRACE("Prepared subgroup impl with factor_wi:", factor_wi, "and factor_sg:", factor_sg);
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
      std::size_t local_memory_usage =
          num_scalars_in_local_mem(detail::level::WORKGROUP, static_cast<std::size_t>(fft_size), SubgroupSize,
                                   {factor_sg_n, factor_wi_n, factor_sg_m, factor_wi_m}, temp_num_sgs_in_wg,
                                   layout::PACKED) *
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
        PORTFFT_LOG_TRACE("Prepared workgroup impl with factor_wi_n:", factor_wi_n, " factor_sg_n:", factor_sg_n,
                          " factor_wi_m:", factor_wi_m, " factor_sg_m:", factor_sg_m);
        return {detail::level::WORKGROUP, {{detail::level::WORKGROUP, ids, factors}}};
      }
    }
    PORTFFT_LOG_TRACE("Preparing global impl");
    std::vector<std::tuple<detail::level, std::vector<sycl::kernel_id>, std::vector<Idx>>> param_vec;
    auto check_and_select_target_level = [&](IdxGlobal factor_size, bool batch_interleaved_layout = true) -> bool {
      if (detail::fits_in_wi<Scalar>(factor_size)) {
        // Throughout we have assumed there would always be enough local memory for the WI implementation.
        param_vec.emplace_back(detail::level::WORKITEM,
                               detail::get_ids<detail::global_kernel, Scalar, Domain, SubgroupSize>(),
                               std::vector<Idx>{static_cast<Idx>(factor_size)});
        PORTFFT_LOG_TRACE("Workitem kernel for factor:", factor_size);
        return true;
      }
      bool fits_in_local_memory_subgroup = [&]() {
        Idx temp_num_sgs_in_wg;
        IdxGlobal factor_sg = detail::factorize_sg<IdxGlobal>(factor_size, SubgroupSize);
        IdxGlobal factor_wi = factor_size / factor_sg;
        if (detail::can_cast_safely<IdxGlobal, Idx>(factor_sg) && detail::can_cast_safely<IdxGlobal, Idx>(factor_wi)) {
          std::size_t input_scalars =
              num_scalars_in_local_mem(detail::level::SUBGROUP, static_cast<std::size_t>(factor_size), SubgroupSize,
                                       {static_cast<Idx>(factor_sg), static_cast<Idx>(factor_wi)}, temp_num_sgs_in_wg,
                                       batch_interleaved_layout ? layout::BATCH_INTERLEAVED : layout::PACKED);
          std::size_t store_modifiers = batch_interleaved_layout ? input_scalars : 0;
          std::size_t twiddle_scalars = 2 * static_cast<std::size_t>(factor_size);
          return (sizeof(Scalar) * (input_scalars + store_modifiers + twiddle_scalars)) <
                 static_cast<std::size_t>(local_memory_size);
        }
        return false;
      }();
      if (detail::fits_in_sg<Scalar>(factor_size, SubgroupSize) && fits_in_local_memory_subgroup &&
          !PORTFFT_SLOW_SG_SHUFFLES) {
        Idx factor_sg = detail::factorize_sg(static_cast<Idx>(factor_size), SubgroupSize);
        Idx factor_wi = static_cast<Idx>(factor_size) / factor_sg;
        PORTFFT_LOG_TRACE("Subgroup kernel for factor:", factor_size, "with factor_wi:", factor_wi,
                          "and factor_sg:", factor_sg);
        param_vec.emplace_back(detail::level::SUBGROUP,
                               detail::get_ids<detail::global_kernel, Scalar, Domain, SubgroupSize>(),
                               std::vector<Idx>{factor_sg, factor_wi});
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
      static void execute(committed_descriptor_impl& desc, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle,
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
   * @param multiply_on_load Whether the input data is multiplied with some data array before fft computation
   * @param multiply_on_store Whether the input data is multiplied with some data array after fft computation
   * @param scale_factor_applied whether or not to multiply scale factor
   * @param level sub implementation to run which will be set as a spec constant
   * @param conjugate_on_load whether or not to take conjugate of the input
   * @param conjugate_on_store whether or not to take conjugate of the output
   * @param scale_factor Scale to be applied to the result
   * @param factor_num factor number which is set as a spec constant
   * @param num_factors total number of factors of the committed size, set as a spec constant
   */
  void set_spec_constants(detail::level top_level, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle,
                          Idx length, const std::vector<Idx>& factors, detail::elementwise_multiply multiply_on_load,
                          detail::elementwise_multiply multiply_on_store,
                          detail::apply_scale_factor scale_factor_applied, detail::level level,
                          detail::complex_conjugate conjugate_on_load, detail::complex_conjugate conjugate_on_store,
                          Scalar scale_factor, IdxGlobal input_stride, IdxGlobal output_stride,
                          IdxGlobal input_distance, IdxGlobal output_distance, Idx factor_num = 0,
                          Idx num_factors = 0) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    // These spec constants are used in all implementations, so we set them here
    PORTFFT_LOG_TRACE("Setting specialization constants:");
    PORTFFT_LOG_TRACE("SpecConstComplexStorage:", params.complex_storage);
    in_bundle.template set_specialization_constant<detail::SpecConstComplexStorage>(params.complex_storage);
    PORTFFT_LOG_TRACE("SpecConstMultiplyOnLoad:", multiply_on_load);
    in_bundle.template set_specialization_constant<detail::SpecConstMultiplyOnLoad>(multiply_on_load);
    PORTFFT_LOG_TRACE("SpecConstMultiplyOnStore:", multiply_on_store);
    in_bundle.template set_specialization_constant<detail::SpecConstMultiplyOnStore>(multiply_on_store);
    PORTFFT_LOG_TRACE("SpecConstApplyScaleFactor:", scale_factor_applied);
    in_bundle.template set_specialization_constant<detail::SpecConstApplyScaleFactor>(scale_factor_applied);
    PORTFFT_LOG_TRACE("SpecConstConjugateOnLoad:", conjugate_on_load);
    in_bundle.template set_specialization_constant<detail::SpecConstConjugateOnLoad>(conjugate_on_load);
    PORTFFT_LOG_TRACE("SpecConstConjugateOnStore:", conjugate_on_store);
    in_bundle.template set_specialization_constant<detail::SpecConstConjugateOnStore>(conjugate_on_store);
    PORTFFT_LOG_TRACE("get_spec_constant_scale:", scale_factor);
    in_bundle.template set_specialization_constant<detail::get_spec_constant_scale<Scalar>()>(scale_factor);
    PORTFFT_LOG_TRACE("SpecConstInputStride:", input_stride);
    in_bundle.template set_specialization_constant<detail::SpecConstInputStride>(input_stride);
    PORTFFT_LOG_TRACE("SpecConstOutputStride:", output_stride);
    in_bundle.template set_specialization_constant<detail::SpecConstOutputStride>(output_stride);
    PORTFFT_LOG_TRACE("SpecConstInputDistance:", input_distance);
    in_bundle.template set_specialization_constant<detail::SpecConstInputDistance>(input_distance);
    PORTFFT_LOG_TRACE("SpecConstOutputDistance:", output_distance);
    in_bundle.template set_specialization_constant<detail::SpecConstOutputDistance>(output_distance);
    dispatch<set_spec_constants_struct>(top_level, in_bundle, length, factors, level, factor_num, num_factors);
  }

  /**
   * Struct for dispatching `num_scalars_in_local_mem()` call.
   */
  struct num_scalars_in_local_mem_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template <detail::level Lev, typename Dummy>
    struct inner {
      static std::size_t execute(committed_descriptor_impl& desc, std::size_t length, Idx used_sg_size,
                                 const std::vector<Idx>& factors, Idx& num_sgs_per_wg, layout input_layout);
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
   * @param input_layout the layout of the input data of the transforms
   * @return the number of scalars
   */
  std::size_t num_scalars_in_local_mem(detail::level level, std::size_t length, Idx used_sg_size,
                                       const std::vector<Idx>& factors, Idx& num_sgs_per_wg, layout input_layout) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    return dispatch<num_scalars_in_local_mem_struct>(level, length, used_sg_size, factors, num_sgs_per_wg,
                                                     input_layout);
  }

  /**
   * Struct for dispatching `calculate_twiddles()` call.
   */
  struct calculate_twiddles_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template <detail::level Lev, typename Dummy>
    struct inner {
      static Scalar* execute(committed_descriptor_impl& desc, dimension_struct& dimension_data,
                             std::vector<kernel_data_struct>& kernels);
    };
  };

  /**
   * Calculates twiddle factors for the implementation in use.
   * @param level Implementation selected for the committed size
   * @param dimension_data dimension_struct correspoding to the dimension for which twiddles are being calculated
   * @param kernels vector of kernels
   * @return Scalar* USM pointer to the twiddle factors
   */
  Scalar* calculate_twiddles(detail::level level, dimension_struct& dimension_data,
                             std::vector<kernel_data_struct>& kernels) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    return dispatch<calculate_twiddles_struct>(level, dimension_data, kernels);
  }

  /**
   * Sets the specialization constants for all the kernel_ids contained in the vector
   * returned from prepare_implementation
   * @tparam SubgroupSize Subgroup size
   * @param top_level selected level of implementation
   * @param prepared_vec vector of tuples of: implementation to use for a kernel,
   * vector of kernel ids, factors
   * @param compute_direction direction of compute: forward or backward
   * @param dimension_num which dimension are the kernels being built for
   * @param skip_scaling whether or not to skip scaling
   * @return vector of kernel_data_struct if all kernel builds are successful, std::nullopt otherwise
   */
  template <Idx SubgroupSize>
  std::optional<std::vector<kernel_data_struct>> set_spec_constants_driver(detail::level top_level,
                                                                           kernel_ids_and_metadata_t& prepared_vec,
                                                                           direction compute_direction,
                                                                           std::size_t dimension_num) {
    Scalar scale_factor = compute_direction == direction::FORWARD ? params.forward_scale : params.backward_scale;
    std::size_t counter = 0;
    IdxGlobal remaining_factors_prod = static_cast<IdxGlobal>(params.get_flattened_length());
    std::vector<kernel_data_struct> result;
    for (auto [level, ids, factors] : prepared_vec) {
      const bool is_multi_dim = params.lengths.size() > 1;
      const bool is_global = top_level == detail::level::GLOBAL;
      const bool is_final_factor = counter == (prepared_vec.size() - 1);
      const bool is_final_dim = dimension_num == (params.lengths.size() - 1);
      const bool is_backward = compute_direction == direction::BACKWARD;
      if (is_multi_dim && is_global) {
        throw unsupported_configuration("multidimensional global transforms are not supported.");
      }

      const auto multiply_on_store = is_global && !is_final_factor ? detail::elementwise_multiply::APPLIED
                                                                   : detail::elementwise_multiply::NOT_APPLIED;
      const auto conjugate_on_load =
          is_backward && counter == 0 ? detail::complex_conjugate::APPLIED : detail::complex_conjugate::NOT_APPLIED;
      const auto conjugate_on_store =
          is_backward && is_final_factor ? detail::complex_conjugate::APPLIED : detail::complex_conjugate::NOT_APPLIED;
      const auto apply_scale = is_final_factor && is_final_dim ? detail::apply_scale_factor::APPLIED
                                                               : detail::apply_scale_factor::NOT_APPLIED;

      Idx length{};
      IdxGlobal forward_stride{};
      IdxGlobal backward_stride{};
      IdxGlobal forward_distance{};
      IdxGlobal backward_distance{};

      if (is_global) {
        length = std::accumulate(factors.begin(), factors.end(), Idx(1), std::multiplies<Idx>());

        remaining_factors_prod /= length;
        forward_stride = remaining_factors_prod;
        backward_stride = remaining_factors_prod;
        forward_distance = is_final_factor ? length : 1;
        backward_distance = is_final_factor ? length : 1;

      } else {
        length = static_cast<Idx>(params.lengths[dimension_num]);
        forward_stride = static_cast<IdxGlobal>(params.forward_strides[dimension_num]);
        backward_stride = static_cast<IdxGlobal>(params.backward_strides[dimension_num]);
        if (is_multi_dim) {
          if (is_final_dim) {
            forward_distance = length;
            backward_distance = length;
          } else {
            forward_distance = 1;
            backward_distance = 1;
          }
        } else {
          forward_distance = static_cast<IdxGlobal>(params.forward_distance);
          backward_distance = static_cast<IdxGlobal>(params.backward_distance);
        }
      }

      const IdxGlobal input_stride = compute_direction == direction::FORWARD ? forward_stride : backward_stride;
      const IdxGlobal output_stride = compute_direction == direction::FORWARD ? backward_stride : forward_stride;
      const IdxGlobal input_distance = compute_direction == direction::FORWARD ? forward_distance : backward_distance;
      const IdxGlobal output_distance = compute_direction == direction::FORWARD ? backward_distance : forward_distance;

      auto in_bundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(queue.get_context(), ids);

      set_spec_constants(top_level, in_bundle, length, factors, detail::elementwise_multiply::NOT_APPLIED,
                         multiply_on_store, apply_scale, level, conjugate_on_load, conjugate_on_store, scale_factor,
                         input_stride, output_stride, input_distance, output_distance, static_cast<Idx>(counter),
                         static_cast<Idx>(prepared_vec.size()));
      try {
        PORTFFT_LOG_TRACE("Building kernel bundle with subgroup size", SubgroupSize);
        result.emplace_back(sycl::build(in_bundle), factors, params.lengths[dimension_num], SubgroupSize,
                            PORTFFT_SGS_IN_WG, std::shared_ptr<Scalar>(), level);
        PORTFFT_LOG_TRACE("Kernel bundle build complete.");
      } catch (std::exception& e) {
        PORTFFT_LOG_WARNING("Build for subgroup size", SubgroupSize, "failed with message:\n", e.what());
        return std::nullopt;
      }
      counter++;
    }
    return result;
  }

  /**
   * Builds the kernel bundles with appropriate values of specialization constants for the first supported subgroup
   * size.
   *
   * @tparam SubgroupSize first subgroup size
   * @tparam OtherSGSizes other subgroup sizes
   * @param dimension_num The dimension for which the kernels are being built
   * @param skip_scaling whether or not to skip scaling
   * @return `dimension_struct` for the newly built kernels
   */
  template <Idx SubgroupSize, Idx... OtherSGSizes>
  dimension_struct build_w_spec_const(std::size_t dimension_num) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    if (std::count(supported_sg_sizes.begin(), supported_sg_sizes.end(), SubgroupSize)) {
      auto [top_level, prepared_vec] = prepare_implementation<SubgroupSize>(dimension_num);
      bool is_compatible = true;
      for (auto [level, ids, factors] : prepared_vec) {
        is_compatible = is_compatible && sycl::is_compatible(ids, dev);
        if (!is_compatible) {
          break;
        }
      }

      if (is_compatible) {
        auto forward_kernels =
            set_spec_constants_driver<SubgroupSize>(top_level, prepared_vec, direction::FORWARD, dimension_num);
        auto backward_kernels =
            set_spec_constants_driver<SubgroupSize>(top_level, prepared_vec, direction::BACKWARD, dimension_num);
        if (forward_kernels.has_value() && backward_kernels.has_value()) {
          return {forward_kernels.value(), backward_kernels.value(), top_level, params.lengths[dimension_num],
                  SubgroupSize};
        }
      }
    }
    if constexpr (sizeof...(OtherSGSizes) == 0) {
      throw unsupported_configuration("None of the compiled subgroup sizes are supported by the device");
    } else {
      return build_w_spec_const<OtherSGSizes...>(dimension_num);
    }
  }

  /**
   * Function which calculates the amount of scratch space required, and also pre computes the necessary scans required.
   * @param num_global_level_dimensions number of global level dimensions in the committed size
   */
  void allocate_scratch_and_precompute_scan(Idx num_global_level_dimensions) {
    PORTFFT_LOG_FUNCTION_ENTRY();
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
      for (const auto& kernel_data : dimensions.at(global_dimension).forward_kernels) {
        IdxGlobal factor_size = static_cast<IdxGlobal>(
            std::accumulate(kernel_data.factors.begin(), kernel_data.factors.end(), 1, std::multiplies<Idx>()));
        cache_required_for_twiddles +=
            static_cast<std::size_t>(2 * factor_size * kernel_data.batch_size) * sizeof(Scalar);
        factors.push_back(factor_size);
        sub_batches.push_back(kernel_data.batch_size);
      }
      dimensions.at(global_dimension).num_factors = static_cast<Idx>(factors.size());
      std::size_t cache_space_left_for_batches = static_cast<std::size_t>(llc_size) - cache_required_for_twiddles;
      // TODO: In case of multi-dim (single dim global sized), this should be batches corresponding to that dim
      dimensions.at(global_dimension).num_batches_in_l2 = static_cast<Idx>(std::min(
          static_cast<std::size_t>(PORTFFT_MAX_CONCURRENT_KERNELS),
          std::min(params.number_of_transforms,
                   std::max(std::size_t(1), cache_space_left_for_batches /
                                                (2 * dimensions.at(global_dimension).length * sizeof(Scalar))))));
      scratch_space_required = 2 * dimensions.at(global_dimension).length *
                               static_cast<std::size_t>(dimensions.at(global_dimension).num_batches_in_l2);
      PORTFFT_LOG_TRACE("Allocating 2 scratch arrays of size", scratch_space_required, "scalars in global memory");
      scratch_ptr_1 = detail::make_shared<Scalar>(scratch_space_required, queue);
      scratch_ptr_2 = detail::make_shared<Scalar>(scratch_space_required, queue);
      inclusive_scan.push_back(factors.at(0));
      for (std::size_t i = 1; i < factors.size(); i++) {
        inclusive_scan.push_back(inclusive_scan.at(i - 1) * factors.at(i));
      }
      PORTFFT_LOG_TRACE("Dimension:", global_dimension,
                        "num_batches_in_l2:", dimensions.at(global_dimension).num_batches_in_l2,
                        "scan:", inclusive_scan);
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
        PORTFFT_LOG_TRACE("Setting specialization constants for transpose kernel", i);
        PORTFFT_LOG_TRACE("SpecConstComplexStorage:", params.complex_storage);
        in_bundle.template set_specialization_constant<detail::SpecConstComplexStorage>(params.complex_storage);
        PORTFFT_LOG_TRACE("GlobalSpecConstLevelNum:", i);
        in_bundle.template set_specialization_constant<detail::GlobalSpecConstLevelNum>(static_cast<Idx>(i));
        PORTFFT_LOG_TRACE("GlobalSpecConstNumFactors:", factors.size());
        in_bundle.template set_specialization_constant<detail::GlobalSpecConstNumFactors>(
            static_cast<Idx>(factors.size()));
        dimensions.at(global_dimension)
            .transpose_kernels.emplace_back(
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
          for (const auto& kernel_data : dimensions.at(i).forward_kernels) {
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
            PORTFFT_LOG_TRACE("Setting specilization constants for transpose kernel", j);
            PORTFFT_LOG_TRACE("GlobalSpecConstLevelNum:", i);
            in_bundle.template set_specialization_constant<detail::GlobalSpecConstLevelNum>(static_cast<Idx>(i));
            PORTFFT_LOG_TRACE("GlobalSpecConstNumFactors:", factors.size());
            in_bundle.template set_specialization_constant<detail::GlobalSpecConstNumFactors>(
                static_cast<Idx>(factors.size()));
            dimensions.at(i).transpose_kernels.emplace_back(
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
  committed_descriptor_impl(const descriptor<Scalar, Domain>& params, sycl::queue& queue)
      : params(params),
        queue(queue),
        dev(queue.get_device()),
        ctx(queue.get_context()),
        // get some properties we will use for tunning
        n_compute_units(static_cast<Idx>(dev.get_info<sycl::info::device::max_compute_units>())),
        supported_sg_sizes(dev.get_info<sycl::info::device::sub_group_sizes>()),
        local_memory_size(static_cast<Idx>(queue.get_device().get_info<sycl::info::device::local_mem_size>())),
        llc_size(static_cast<IdxGlobal>(queue.get_device().get_info<sycl::info::device::global_mem_cache_size>())) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    PORTFFT_LOG_TRACE("Device info:");
    PORTFFT_LOG_TRACE("n_compute_units:", n_compute_units);
    PORTFFT_LOG_TRACE("supported_sg_sizes:", supported_sg_sizes);
    PORTFFT_LOG_TRACE("local_memory_size:", local_memory_size);
    PORTFFT_LOG_TRACE("llc_size:", llc_size);

    // compile the kernels and precalculate twiddles
    std::size_t n_kernels = params.lengths.size();
    for (std::size_t i = 0; i < n_kernels; i++) {
      dimensions.emplace_back(build_w_spec_const<PORTFFT_SUBGROUP_SIZES>(i));
      dimensions.back().forward_kernels.at(0).twiddles_forward = std::shared_ptr<Scalar>(
          calculate_twiddles(dimensions.back().level, dimensions.at(i), dimensions.back().forward_kernels),
          [queue](Scalar* ptr) {
            if (ptr != nullptr) {
              sycl::free(ptr, queue);
            }
          });
      // TODO: refactor multi-dimensional fft's such that they can use a single pointer for twiddles.
      dimensions.back().backward_kernels.at(0).twiddles_forward = std::shared_ptr<Scalar>(
          calculate_twiddles(dimensions.back().level, dimensions.at(i), dimensions.back().backward_kernels),
          [queue](Scalar* ptr) {
            if (ptr != nullptr) {
              PORTFFT_LOG_TRACE("Freeing the array for twiddle factors");
              sycl::free(ptr, queue);
            }
          });
    }

    Idx num_global_level_dimensions = static_cast<Idx>(std::count_if(
        dimensions.cbegin(), dimensions.cend(), [](auto& d) { return d.level == detail::level::GLOBAL; }));
    if (num_global_level_dimensions != 0) {
      if (params.lengths.size() > 1) {
        throw unsupported_configuration("For FFTs that do not fit in local memory only 1D is supported");
      }
      if (params.get_distance(direction::FORWARD) != params.lengths[0] ||
          params.get_distance(direction::BACKWARD) != params.lengths[0]) {
        throw unsupported_configuration("Large FFTs are currently only supported in non-strided format");
      }

      allocate_scratch_and_precompute_scan(num_global_level_dimensions);
    }
  }

  /**
   * Utility function for copy constructor and copy assignment operator
   * @param desc `committed_descriptor_impl` of which the copy is to be made
   */
  void create_copy(const committed_descriptor_impl<Scalar, Domain>& desc) {
    PORTFFT_LOG_FUNCTION_ENTRY();
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
      PORTFFT_LOG_TRACE("Allocating 2 scratch arrays of size", desc.scratch_space_required, "Scalars in global memory");
      this->scratch_ptr_1 =
          detail::make_shared<Scalar>(static_cast<std::size_t>(desc.scratch_space_required), this->queue);
      this->scratch_ptr_2 =
          detail::make_shared<Scalar>(static_cast<std::size_t>(desc.scratch_space_required), this->queue);
    }
  }

 public:
  committed_descriptor_impl(const committed_descriptor_impl& desc) : params(desc.params) {  // TODO params copied twice
    PORTFFT_LOG_FUNCTION_ENTRY();
    create_copy(desc);
  }

  committed_descriptor_impl& operator=(const committed_descriptor_impl& desc) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    if (this != &desc) {
      create_copy(desc);
    }
    return *this;
  }

  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "Scalar must be either float or double!");

  /**
   * Destructor
   */
  ~committed_descriptor_impl() {
    PORTFFT_LOG_FUNCTION_ENTRY();
    queue.wait();
  }

  // default construction is not appropriate
  committed_descriptor_impl() = delete;

 protected:
  /**
   * Dispatches to the implementation for the appropriate direction.
   *
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
   * @param compute_direction direction of compute, forward / backward
   * @param dependencies events that must complete before the computation
   * @return sycl::event
   */
  template <typename TIn, typename TOut>
  sycl::event dispatch_direction(const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                                 complex_storage used_storage, direction compute_direction,
                                 const std::vector<sycl::event>& dependencies = {}) {
    PORTFFT_LOG_FUNCTION_ENTRY();
#ifndef PORTFFT_ENABLE_BUFFER_BUILDS
    if constexpr (!std::is_pointer_v<TIn> || !std::is_pointer_v<TOut>) {
      throw invalid_configuration("Buffer interface can not be called when buffer builds are disabled.");
    }
#endif
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
    if (compute_direction == direction::FORWARD) {
      return dispatch_dimensions(in, out, in_imag, out_imag, dependencies, params.forward_offset,
                                 params.backward_offset, compute_direction);
    }
    return dispatch_dimensions(in, out, in_imag, out_imag, dependencies, params.backward_offset, params.forward_offset,
                               compute_direction);
  }

  /**
   * Dispatches to the implementation for the appropriate number of dimensions.
   *
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
   * @param input_offset offset into input allocation where the data for FFTs start
   * @param output_offset offset into output allocation where the data for FFTs start
   * @param compute_direction direction of compute, forward / backward
   * @return sycl::event
   */
  template <typename TIn, typename TOut>
  sycl::event dispatch_dimensions(const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                                  const std::vector<sycl::event>& dependencies, std::size_t input_offset,
                                  std::size_t output_offset, direction compute_direction) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    using TOutConst = std::conditional_t<std::is_pointer_v<TOut>, const std::remove_pointer_t<TOut>*, const TOut>;
    std::size_t n_dimensions = params.lengths.size();
    std::size_t total_size = params.get_flattened_length();

    const auto input_layout = detail::get_layout(params, compute_direction);
    const auto output_layout = detail::get_layout(params, inv(compute_direction));

    // currently multi-dimensional transforms are implemented just for default (PACKED) data layout
    const bool multi_dim_supported = input_layout == detail::layout::PACKED && output_layout == detail::layout::PACKED;
    if (n_dimensions != 1 && !multi_dim_supported) {
      throw internal_error("Only default layout is supported for multi-dimensional transforms.");
    }

    // product of sizes of all dimension inner relative to the one we are currently working on
    std::size_t inner_size = 1;
    // product of sizes of all dimension outer relative to the one we are currently working on
    std::size_t outer_size = total_size / params.lengths.back();

    PORTFFT_LOG_TRACE("Dispatching the kernel for the last dimension");
    sycl::event previous_event =
        dispatch_kernel_1d(in, out, in_imag, out_imag, dependencies, params.number_of_transforms * outer_size,
                           input_layout, input_offset, output_offset, dimensions.back(), compute_direction);
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
      PORTFFT_LOG_TRACE("Dispatching the kernels for the dimension", i);
      for (std::size_t j = 0; j < params.number_of_transforms * outer_size; j++) {
        sycl::event e = dispatch_kernel_1d<TOutConst, TOut>(
            out, out, out_imag, out_imag, previous_events, inner_size, layout::BATCH_INTERLEAVED,
            output_offset + j * stride_between_kernels, output_offset + j * stride_between_kernels, dimensions[i],
            compute_direction);
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
   * @param input_layout the layout of the input data of the transforms
   * @param input_offset offset into input allocation where the data for FFTs start
   * @param output_offset offset into output allocation where the data for FFTs start
   * @param dimension_data data for the dimension this call will work on
   * @param compute_direction direction of compute, forward / backward
   * @return sycl::event
   */
  template <typename TIn, typename TOut>
  sycl::event dispatch_kernel_1d(const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                                 const std::vector<sycl::event>& dependencies, std::size_t n_transforms,
                                 layout input_layout, std::size_t input_offset, std::size_t output_offset,
                                 dimension_struct& dimension_data, direction compute_direction) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    return dispatch_kernel_1d_helper<TIn, TOut, PORTFFT_SUBGROUP_SIZES>(
        in, out, in_imag, out_imag, dependencies, n_transforms, input_layout, input_offset, output_offset,
        dimension_data, compute_direction);
  }

  /**
   * Helper for dispatching the kernel with the first subgroup size that is supported by the device.
   *
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
   * @param input_layout the layout of the input data of the transforms
   * @param input_offset offset into input allocation where the data for FFTs start
   * @param output_offset offset into output allocation where the data for FFTs start
   * @param dimension_data data for the dimension this call will work on
   * @param compute_direction direction of compute, forward / backward
   * @return sycl::event
   */
  template <typename TIn, typename TOut, Idx SubgroupSize, Idx... OtherSGSizes>
  sycl::event dispatch_kernel_1d_helper(const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                                        const std::vector<sycl::event>& dependencies, std::size_t n_transforms,
                                        layout input_layout, std::size_t input_offset, std::size_t output_offset,
                                        dimension_struct& dimension_data, direction compute_direction) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    if (SubgroupSize == dimension_data.used_sg_size) {
      const bool input_batch_interleaved = input_layout == layout::BATCH_INTERLEAVED;

      for (kernel_data_struct kernel_data : dimension_data.forward_kernels) {
        if (input_batch_interleaved) {
          std::size_t minimum_local_mem_required =
              num_scalars_in_local_mem(kernel_data.level, kernel_data.length, SubgroupSize, kernel_data.factors,
                                       kernel_data.num_sgs_per_wg, layout::BATCH_INTERLEAVED) *
              sizeof(Scalar);
          PORTFFT_LOG_TRACE("Local mem required:", minimum_local_mem_required, "B. Available: ", local_memory_size,
                            "B.");
          if (static_cast<Idx>(minimum_local_mem_required) > local_memory_size) {
            throw out_of_local_memory_error(
                "Insufficient amount of local memory available: " + std::to_string(local_memory_size) +
                "B. Required: " + std::to_string(minimum_local_mem_required) + "B.");
          }
        }
      }

      return run_kernel<SubgroupSize>(in, out, in_imag, out_imag, dependencies, n_transforms, input_offset,
                                      output_offset, dimension_data, compute_direction, input_layout);
    }
    if constexpr (sizeof...(OtherSGSizes) == 0) {
      throw invalid_configuration("None of the compiled subgroup sizes are supported by the device!");
    } else {
      return dispatch_kernel_1d_helper<TIn, TOut, OtherSGSizes...>(in, out, in_imag, out_imag, dependencies,
                                                                   n_transforms, input_layout, input_offset,
                                                                   output_offset, dimension_data, compute_direction);
    }
  }

  /**
   * Struct for dispatching `run_kernel()` call.
   *
   * @tparam SubgroupSize size of the subgroup
   * @tparam TIn Type of the input USM pointer or buffer
   * @tparam TOut Type of the output USM pointer or buffer
   */
  template <Idx SubgroupSize, typename TIn, typename TOut>
  struct run_kernel_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template <detail::level Lev, typename Dummy>
    struct inner {
      static sycl::event execute(committed_descriptor_impl& desc, const TIn& in, TOut& out, const TIn& in_imag,
                                 TOut& out_imag, const std::vector<sycl::event>& dependencies, std::size_t n_transforms,
                                 std::size_t forward_offset, std::size_t backward_offset,
                                 dimension_struct& dimension_data, direction compute_direction, layout input_layout);
    };
  };

  /**
   * Common interface to run the kernel called by compute_forward and compute_backward
   *
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
   * @param dimension_data data for the dimension this call will work on
   * @param compute_direction direction of fft, forward / backward
   * @param input_layout the layout of the input data of the transforms
   * @return sycl::event
   */
  template <Idx SubgroupSize, typename TIn, typename TOut>
  sycl::event run_kernel(const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                         const std::vector<sycl::event>& dependencies, std::size_t n_transforms,
                         std::size_t input_offset, std::size_t output_offset, dimension_struct& dimension_data,
                         direction compute_direction, layout input_layout) {
    PORTFFT_LOG_FUNCTION_ENTRY();
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
    return dispatch<run_kernel_struct<SubgroupSize, TInReinterpret, TOutReinterpret>>(
        dimension_data.level, detail::reinterpret<const Scalar>(in), detail::reinterpret<Scalar>(out),
        detail::reinterpret<const Scalar>(in_imag), detail::reinterpret<Scalar>(out_imag), dependencies,
        static_cast<IdxGlobal>(n_transforms), static_cast<IdxGlobal>(vec_multiplier * input_offset),
        static_cast<IdxGlobal>(vec_multiplier * output_offset), dimension_data, compute_direction, input_layout);
  }
};

}  // namespace detail
}  // namespace portfft

#endif
