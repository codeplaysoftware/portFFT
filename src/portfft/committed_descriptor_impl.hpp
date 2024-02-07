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
template <typename Scalar, domain Domain>
using kernels_vec = std::vector<typename committed_descriptor_impl<Scalar, Domain>::kernel_data_struct>;

template <typename Scalar, domain Domain, detail::layout LayoutIn, detail::layout LayoutOut, Idx SubgroupSize,
          typename TIn>
std::vector<sycl::event> compute_level(const typename committed_descriptor_impl<Scalar, Domain>::kernel_data_struct&,
                                       const TIn&, Scalar*, const TIn&, Scalar*, const Scalar*, const Scalar*,
                                       const Scalar*, const IdxGlobal*, IdxGlobal, IdxGlobal, Idx, IdxGlobal, IdxGlobal,
                                       Idx, Idx, complex_storage, const std::vector<sycl::event>&, sycl::queue&);

template <typename Scalar, domain Domain, typename TOut>
sycl::event transpose_level(const typename committed_descriptor_impl<Scalar, Domain>::kernel_data_struct&,
                            const Scalar*, TOut, const IdxGlobal*, IdxGlobal, Idx, IdxGlobal, IdxGlobal, Idx, IdxGlobal,
                            sycl::queue&, const std::vector<sycl::event>&, complex_storage);

template <Idx, typename Scalar, domain Domain, typename TIn, typename TOut>
sycl::event global_impl_driver(const TIn&, const TIn&, TOut, TOut, committed_descriptor_impl<Scalar, Domain>&,
                               typename committed_descriptor_impl<Scalar, Domain>::dimension_struct&,
                               const kernels_vec<Scalar, Domain>&, const kernels_vec<Scalar, Domain>&, Idx, IdxGlobal,
                               IdxGlobal, std::size_t, std::size_t, IdxGlobal, IdxGlobal, IdxGlobal, complex_storage,
                               detail::elementwise_multiply, const Scalar*);

// kernel names
// TODO: Remove all templates except Scalar, Domain and Memory and SubgroupSize
template <typename Scalar, domain, detail::memory, detail::layout, detail::layout, Idx SubgroupSize>
class workitem_kernel;
template <typename Scalar, domain, detail::memory, detail::layout, detail::layout, Idx SubgroupSize>
class subgroup_kernel;
template <typename Scalar, domain, detail::memory, detail::layout, detail::layout, Idx SubgroupSize>
class workgroup_kernel;
template <typename Scalar, domain, detail::memory, detail::layout, detail::layout, Idx SubgroupSize>
class global_kernel;
template <typename Scalar, detail::memory>
class transpose_kernel;

/**
 * Return the default strides for a given dft size
 *
 * @param lengths the dimensions of the dft
 */
inline std::vector<std::size_t> get_default_strides(const std::vector<std::size_t>& lengths) {
  PORTFFT_LOG_FUNCTION_ENTRY();
  std::vector<std::size_t> strides(lengths.size());
  std::size_t total_size = 1;
  for (std::size_t i_plus1 = lengths.size(); i_plus1 > 0; i_plus1--) {
    std::size_t i = i_plus1 - 1;
    strides[i] = total_size;
    total_size *= lengths[i];
  }
  PORTFFT_LOG_TRACE("Default strides:", strides);
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

/**
 * A committed descriptor that contains everything that is needed to run FFT.
 *
 * @tparam Scalar type of the scalar used for computations
 * @tparam Domain domain of the FFT
 */
template <typename Scalar, domain Domain>
class committed_descriptor_impl {
  friend struct descriptor<Scalar, Domain>;
  template <typename Scalar1, domain Domain1, detail::layout LayoutIn, detail::layout LayoutOut, Idx SubgroupSize,
            typename TIn>
  friend std::vector<sycl::event> compute_level(
      const typename committed_descriptor_impl<Scalar1, Domain1>::kernel_data_struct&, const TIn&, Scalar1*, const TIn&,
      Scalar1*, const Scalar1*, const Scalar1*, const Scalar1*, const IdxGlobal*, IdxGlobal, IdxGlobal, Idx, IdxGlobal,
      IdxGlobal, Idx, Idx, complex_storage, const std::vector<sycl::event>&, sycl::queue&);

  template <typename Scalar1, domain Domain1, typename TOut>
  friend sycl::event detail::transpose_level(
      const typename committed_descriptor_impl<Scalar1, Domain1>::kernel_data_struct&, const Scalar1*, TOut,
      const IdxGlobal*, IdxGlobal, Idx, IdxGlobal, IdxGlobal, Idx, IdxGlobal, sycl::queue&,
      const std::vector<sycl::event>&, complex_storage);

  template <Idx, typename Scalar1, domain Domain1, typename TIn, typename TOut>
  friend sycl::event global_impl_driver(const TIn&, const TIn&, TOut, TOut,
                                        committed_descriptor_impl<Scalar1, Domain1>&,
                                        typename committed_descriptor_impl<Scalar1, Domain1>::dimension_struct&,
                                        const kernels_vec<Scalar1, Domain1>&, const kernels_vec<Scalar1, Domain1>&, Idx,
                                        IdxGlobal, IdxGlobal, std::size_t, std::size_t, IdxGlobal, IdxGlobal, IdxGlobal,
                                        complex_storage, detail::elementwise_multiply, const Scalar1*);

  /**
   * Vector containing the sub-implementation level, kernel_ids and factors for each factor that requires a separate
   * kernel.
   */
  using kernel_ids_and_metadata_t =
      std::vector<std::tuple<detail::level, std::vector<sycl::kernel_id>, std::vector<Idx>>>;
  /**
   * Tuple of the level, an input kernel_bundle, and factors pertaining to each factor of the committed size
   */
  using input_bundles_and_metadata_t =
      std::tuple<detail::level, sycl::kernel_bundle<sycl::bundle_state::input>, std::vector<Idx>>;

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
    std::size_t committed_length;
    Idx used_sg_size;
    Idx num_batches_in_l2;
    Idx num_forward_factors;
    Idx num_backward_factors;
    bool is_prime;
    IdxGlobal backward_twiddles_offset;
    IdxGlobal bluestein_modifiers_offset;
    IdxGlobal forward_impl_twiddle_offset;
    IdxGlobal backward_impl_twiddle_offset;

    dimension_struct(std::vector<kernel_data_struct> forward_kernels, std::vector<kernel_data_struct> backward_kernels,
                     detail::level level, std::size_t length, std::size_t committed_length, Idx used_sg_size,
                     Idx num_forward_factors, Idx num_backward_factors, bool is_prime)
        : forward_kernels(std::move(forward_kernels)),
          backward_kernels(std::move(backward_kernels)),
          level(level),
          length(length),
          committed_length(committed_length),
          used_sg_size(used_sg_size),
          num_forward_factors(num_forward_factors),
          num_backward_factors(num_backward_factors),
          is_prime(is_prime) {}
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
  std::tuple<detail::level, std::size_t, kernel_ids_and_metadata_t> prepare_implementation(std::size_t kernel_num) {
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
      return {detail::level::WORKITEM, static_cast<std::size_t>(fft_size), {{detail::level::WORKITEM, ids, factors}}};
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
      return {detail::level::SUBGROUP, static_cast<std::size_t>(fft_size), {{detail::level::SUBGROUP, ids, factors}}};
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
        PORTFFT_LOG_TRACE("Prepared workgroup impl with factor_wi_n:", factor_wi_n, " factor_sg_n:", factor_sg_n,
                          " factor_wi_m:", factor_wi_m, " factor_sg_m:", factor_sg_m);
        return {
            detail::level::WORKGROUP, static_cast<std::size_t>(fft_size), {{detail::level::WORKGROUP, ids, factors}}};
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
    if (!detail::factorize_input(fft_size, check_and_select_target_level)) {
      param_vec.clear();
      fft_size = static_cast<IdxGlobal>(std::pow(2, ceil(log(static_cast<double>(fft_size)) / log(2.0))));
      detail::factorize_input(fft_size, check_and_select_target_level);
      detail::factorize_input(fft_size, check_and_select_target_level);
    }
    return {detail::level::GLOBAL, static_cast<std::size_t>(fft_size), param_vec};
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
                          std::size_t length, const std::vector<Idx>& factors,
                          detail::elementwise_multiply multiply_on_load, detail::elementwise_multiply multiply_on_store,
                          detail::apply_scale_factor scale_factor_applied, detail::level level,
                          detail::complex_conjugate conjugate_on_load, detail::complex_conjugate conjugate_on_store,
                          Scalar scale_factor, Idx factor_num = 0, Idx num_factors = 0) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    const Idx length_idx = static_cast<Idx>(length);
    // These spec constants are used in all implementations, so we set them here
    PORTFFT_LOG_TRACE("Setting specialization constants:");
    PORTFFT_LOG_TRACE("SpecConstComplexStorage:", params.complex_storage);
    in_bundle.template set_specialization_constant<detail::SpecConstComplexStorage>(params.complex_storage);
    PORTFFT_LOG_TRACE("SpecConstNumRealsPerFFT:", 2 * length_idx);
    in_bundle.template set_specialization_constant<detail::SpecConstNumRealsPerFFT>(2 * length_idx);
    PORTFFT_LOG_TRACE("SpecConstWIScratchSize:", 2 * detail::wi_temps(length_idx));
    in_bundle.template set_specialization_constant<detail::SpecConstWIScratchSize>(2 * detail::wi_temps(length_idx));
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

    dispatch<set_spec_constants_struct>(top_level, in_bundle, length, factors, level, factor_num, num_factors);
  }

  /**
   * Struct for dispatching `num_scalars_in_local_mem()` call.
   */
  struct num_scalars_in_local_mem_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template <detail::level Lev, detail::layout LayoutIn, typename Dummy>
    struct inner {
      static std::size_t execute(committed_descriptor_impl& desc, std::size_t length, Idx used_sg_size,
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
    PORTFFT_LOG_FUNCTION_ENTRY();
    return dispatch<num_scalars_in_local_mem_struct, LayoutIn>(level, length, used_sg_size, factors, num_sgs_per_wg);
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
   * Sets the specialization constants for the global implementation
   * @param input_kernels_and_metadata vector of input_bundles_and_metadata_t
   * @param num_forward_factors  Number of forward factors
   * @param num_backward_factors Number of backward factors
   * @param compute_direction direction of compute: forward / backward
   * @param is_prime whether or not the dimension size is a prime number
   * @param scale_factor Scaling factor to be applied to the result
   */
  void set_global_impl_spec_constants(std::vector<input_bundles_and_metadata_t>& input_kernels_and_metadata,
                                      Idx num_forward_factors, Idx num_backward_factors, direction compute_direction,
                                      bool is_prime, Scalar scale_factor) {
    detail::complex_conjugate conjugate_on_load;
    detail::complex_conjugate conjugate_on_store;
    detail::elementwise_multiply multiply_on_load;
    detail::elementwise_multiply multiply_on_store;
    detail::apply_scale_factor scale_factor_applied;

    for (std::size_t i = 0; i < std::size_t(num_forward_factors); i++) {
      conjugate_on_load = detail::complex_conjugate::NOT_APPLIED;
      conjugate_on_store = detail::complex_conjugate::NOT_APPLIED;
      multiply_on_load = detail::elementwise_multiply::NOT_APPLIED;
      multiply_on_store = detail::elementwise_multiply::APPLIED;
      scale_factor_applied = detail::apply_scale_factor::NOT_APPLIED;

      if (i == 0 && compute_direction == direction::BACKWARD) {
        conjugate_on_load = detail::complex_conjugate::APPLIED;
      }
      if (i == 0 && is_prime) {
        multiply_on_load = detail::elementwise_multiply::APPLIED;
      }
      if (i == std::size_t(num_forward_factors - 1)) {
        if (compute_direction == direction::BACKWARD && !is_prime) {
          conjugate_on_store = detail::complex_conjugate::APPLIED;
        }
        if (!is_prime) {
          multiply_on_store = detail::elementwise_multiply::NOT_APPLIED;
        }
        if (!is_prime) {
          scale_factor_applied = detail::apply_scale_factor::APPLIED;
        }
      }
      auto& [level, input_bundle, factors] = input_kernels_and_metadata.at(i);
      set_spec_constants(
          detail::level::GLOBAL, input_bundle,
          static_cast<std::size_t>(std::accumulate(factors.begin(), factors.end(), 1, std::multiplies<Idx>())), factors,
          multiply_on_load, multiply_on_store, scale_factor_applied, level, conjugate_on_load, conjugate_on_store,
          scale_factor, Idx(i), num_forward_factors);
    }

    for (std::size_t i = 0; i < std::size_t(num_backward_factors); i++) {
      conjugate_on_load = detail::complex_conjugate::NOT_APPLIED;
      conjugate_on_store = detail::complex_conjugate::NOT_APPLIED;
      multiply_on_load = detail::elementwise_multiply::NOT_APPLIED;
      multiply_on_store = detail::elementwise_multiply::APPLIED;
      scale_factor_applied = detail::apply_scale_factor::NOT_APPLIED;
      if (i == 0) {
        conjugate_on_load = detail::complex_conjugate::APPLIED;
      }
      if (i == std::size_t(num_forward_factors - 1)) {
        multiply_on_store = detail::elementwise_multiply::APPLIED;
        scale_factor_applied = detail::apply_scale_factor::APPLIED;
      }
      auto& [level, input_bundle, factors] = input_kernels_and_metadata.at(std::size_t(num_forward_factors) + i);
      set_spec_constants(
          detail::level::GLOBAL, input_bundle,
          static_cast<std::size_t>(std::accumulate(factors.begin(), factors.end(), 1, std::multiplies<Idx>())), factors,
          multiply_on_load, multiply_on_store, scale_factor_applied, level, conjugate_on_load, conjugate_on_store,
          scale_factor, Idx(i), num_forward_factors);
    }
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
  std::optional<std::vector<kernel_data_struct>> set_spec_constants_driver(
      detail::level top_level, kernel_ids_and_metadata_t& prepared_vec, direction compute_direction,
      std::size_t dimension_num, bool skip_scaling, Idx num_forward_factors, Idx num_backward_factors) {
    Scalar scale_factor = compute_direction == direction::FORWARD ? params.forward_scale : params.backward_scale;
    std::vector<kernel_data_struct> result;
    std::vector<input_bundles_and_metadata_t> input_kernels_and_metadata;
    for (const auto& [level, kernel_ids, factors] : prepared_vec) {
      input_kernels_and_metadata.emplace_back(
          level, sycl::get_kernel_bundle<sycl::bundle_state::input>(queue.get_context(), kernel_ids), factors);
    }
    bool is_compatible = true;
    if (top_level == detail::level::GLOBAL) {
      set_global_impl_spec_constants(input_kernels_and_metadata, num_forward_factors, num_backward_factors,
                                     compute_direction, num_backward_factors > 0, scale_factor);
    } else {
      detail::complex_conjugate conjugate_on_load = detail::complex_conjugate::NOT_APPLIED;
      detail::complex_conjugate conjugate_on_store = detail::complex_conjugate::NOT_APPLIED;
      detail::apply_scale_factor scale_factor_applied = detail::apply_scale_factor::APPLIED;
      if (compute_direction == direction::BACKWARD) {
        conjugate_on_load = detail::complex_conjugate::APPLIED;
        conjugate_on_store = detail::complex_conjugate::APPLIED;
      }
      if (skip_scaling) {
        scale_factor_applied = detail::apply_scale_factor::NOT_APPLIED;
      }
      for (auto& [level, input_bundle, factors] : input_kernels_and_metadata) {
        set_spec_constants(level, input_bundle, params.lengths[dimension_num], factors,
                           detail::elementwise_multiply::NOT_APPLIED, detail::elementwise_multiply::NOT_APPLIED,
                           scale_factor_applied, level, conjugate_on_load, conjugate_on_store, scale_factor);
      }
    }

    for (const auto& [level, input_bundle, factors] : input_kernels_and_metadata) {
      try {
        result.emplace_back(
            sycl::build(input_bundle), factors,
            static_cast<std::size_t>(std::accumulate(factors.begin(), factors.end(), 1, std::multiplies<Idx>())),
            SubgroupSize, PORTFFT_SGS_IN_WG, std::shared_ptr<Scalar>(), level);
      } catch (const std::exception& e) {
        PORTFFT_LOG_WARNING("Build for subgroup size", SubgroupSize, "failed with message:\n", e.what());
        is_compatible = false;
      }
      if (!is_compatible) {
        return std::nullopt;
      }
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
  dimension_struct build_w_spec_const(std::size_t dimension_num, bool skip_scaling) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    if (std::count(supported_sg_sizes.begin(), supported_sg_sizes.end(), SubgroupSize)) {
      auto [top_level, dimension_size, prepared_vec] = prepare_implementation<SubgroupSize>(dimension_num);
      bool is_compatible = true;
      std::size_t temp = 1;
      Idx num_forward_factors = 0;
      for (const auto& [level, ids, factors] : prepared_vec) {
        is_compatible = is_compatible && sycl::is_compatible(ids, dev);
        if (!is_compatible) {
          break;
        }
        if (temp == dimension_size) {
          break;
        }
        temp *= static_cast<std::size_t>(std::accumulate(factors.begin(), factors.end(), 1, std::multiplies<Idx>()));
        num_forward_factors++;
      }
      Idx num_backward_factors = static_cast<Idx>(prepared_vec.size()) - num_forward_factors;
      bool is_prime = static_cast<bool>(dimension_size != params.lengths[dimension_num]);
      if (is_compatible) {
        auto forward_kernels =
            set_spec_constants_driver<SubgroupSize>(top_level, prepared_vec, direction::FORWARD, dimension_num,
                                                    skip_scaling, num_forward_factors, num_backward_factors);
        auto backward_kernels =
            set_spec_constants_driver<SubgroupSize>(top_level, prepared_vec, direction::BACKWARD, dimension_num,
                                                    skip_scaling, num_forward_factors, num_backward_factors);
        if (forward_kernels.has_value() && backward_kernels.has_value()) {
          return {forward_kernels.value(), backward_kernels.value(),      top_level,
                  dimension_size,          params.lengths[dimension_num], SubgroupSize,
                  num_forward_factors,     num_backward_factors,          is_prime};
        }
      }
    }
    if constexpr (sizeof...(OtherSGSizes) == 0) {
      throw invalid_configuration("None of the compiled subgroup sizes are supported by the device");
    } else {
      return build_w_spec_const<OtherSGSizes...>(dimension_num, skip_scaling);
    }
  }

  /**
   * Builds transpose kernels required for global implementation
   * @param dimension_data dimension_struct associated with the dimension
   * @param num_transpositions Number of transpose kernels to build
   * @param ld_input vector containing leading dimensions of the inputs
   * @param ld_output vector containing leading dimensions of the outputs
   */
  void build_transpose_kernels(dimension_struct& dimension_data, std::size_t num_transpositions,
                               std::vector<IdxGlobal>& ld_input, std::vector<IdxGlobal>& ld_output) {
    for (std::size_t i = 0; i < num_transpositions; i++) {
      auto in_bundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(queue.get_context(),
                                                                          detail::get_transpose_kernel_ids<Scalar>());
      in_bundle.template set_specialization_constant<detail::GlobalSpecConstLevelNum>(static_cast<Idx>(i));
      in_bundle.template set_specialization_constant<detail::GlobalSpecConstNumFactors>(
          static_cast<Idx>(num_transpositions + 1));
      try {
        dimension_data.transpose_kernels.emplace_back(
            sycl::build(in_bundle),
            std::vector<Idx>{static_cast<Idx>(ld_input.at(i)), static_cast<Idx>(ld_output.at(i))}, 1, 1, 1,
            std::shared_ptr<Scalar>(), detail::level::GLOBAL);
      } catch (const std::exception& e) {
        throw internal_error("Error building transpose kernel: ", e.what());
      }
    }
  }

  /**
   * Precomputes the inclusive scan required for the global implementation, and populates the device pointer containing
   * the same. Also calculates the ideal amount of llc cache size occupied by the load/store modifiers and returns the
   * same.
   * @param dimension_data Dimension struct for which the inclusive scan is being precomputed
   * @param num_factors Number of factors
   * @param kernel_offset Index from which the kernel_data_struct are to be considered
   * @param ptr Pointer to the global memory for the precomputed data.
   * @return cache space in number of bytes required for the load/store modifiers
   */
  IdxGlobal precompute_scan_impl(dimension_struct& dimension_data, std::size_t num_factors, std::size_t kernel_offset,
                                 IdxGlobal* ptr) {
    std::vector<IdxGlobal> factors;
    std::vector<IdxGlobal> inner_batches;
    std::vector<IdxGlobal> inclusive_scan;

    for (std::size_t i = 0; i < num_factors; i++) {
      const auto& kernel_data = dimension_data.forward_kernels.at(kernel_offset + i);
      factors.push_back(static_cast<IdxGlobal>(kernel_data.length));
      inner_batches.push_back(kernel_data.batch_size);
    }

    inclusive_scan.push_back(factors.at(0));
    for (std::size_t i = 1; i < static_cast<std::size_t>(num_factors); i++) {
      inclusive_scan.push_back(inclusive_scan.at(i - 1) * factors.at(i));
    }
    queue.copy(factors.data(), ptr, factors.size());
    queue.copy(inner_batches.data(), ptr + factors.size(), inner_batches.size());
    queue.copy(inclusive_scan.data(), ptr + factors.size() + inner_batches.size(), inclusive_scan.size());

    build_transpose_kernels(dimension_data, num_factors - 1, factors, inner_batches);

    // calculate Ideal amount of llc cache required for load/store
    std::size_t llc_cache_space_for_twiddles = 0;
    for (std::size_t i = 0; i < num_factors - 1; i++) {
      llc_cache_space_for_twiddles +=
          static_cast<std::size_t>(2 * factors.at(i) * inner_batches.at(i)) * sizeof(Scalar);
    }

    if (dimension_data.is_prime) {
      llc_cache_space_for_twiddles += 4 * dimension_data.length * sizeof(Scalar);
    }
    queue.wait();
    return static_cast<IdxGlobal>(llc_cache_space_for_twiddles);
  }

  /**
   * Gets the number of transforms to accomodate in the last level cache
   * @param llc_cache_space_for_twiddles Amount of cache space in bytes required for load/store modifiers
   * @param n_transforms The Batch size correspoding to the dimension size
   * @param length length of the transform
   * @return
   */
  Idx get_num_batches_in_llc(IdxGlobal llc_cache_space_for_twiddles, IdxGlobal n_transforms, std::size_t length) {
    IdxGlobal cache_space_remaining =
        std::max(IdxGlobal(0), static_cast<IdxGlobal>(llc_size) - llc_cache_space_for_twiddles);
    IdxGlobal sizeof_one_transform = static_cast<IdxGlobal>(2 * length * sizeof(Scalar));

    return static_cast<Idx>(
        std::min(IdxGlobal(PORTFFT_MAX_CONCURRENT_KERNELS),
                 std::min(n_transforms, std::max(IdxGlobal(1), cache_space_remaining / sizeof_one_transform))));
  }

  /**
   * Function which calculates the amount of scratch space required, and also pre computes the necessary scans required.
   * Builds the transpose kernels required for the global implementation
   * @param num_global_level_dimensions number of global level dimensions in the committed size
   */
  void allocate_scratch_and_precompute_scan(Idx num_global_level_dimensions) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    std::size_t n_dimensions = params.lengths.size();
    if (num_global_level_dimensions == 1) {
      std::size_t global_dimension = 0;
      for (std::size_t i = 0; i < n_dimensions; i++) {
        if (dimensions.at(i).level == detail::level::GLOBAL) {
          global_dimension = i;
          break;
        }
      }
      auto& dimension_data = dimensions.at(global_dimension);
      std::size_t space_for_scans =
          static_cast<std::size_t>(3 * (dimension_data.num_forward_factors +
                                        (dimension_data.is_prime ? dimension_data.num_backward_factors : 0)));
      dimension_data.factors_and_scan = detail::make_shared<IdxGlobal>(space_for_scans, queue);
      IdxGlobal cache_req_for_modifiers = static_cast<IdxGlobal>(
          precompute_scan_impl(dimension_data, static_cast<std::size_t>(dimension_data.num_forward_factors), 0,
                               dimension_data.factors_and_scan.get()));
      Idx num_batches_in_llc = get_num_batches_in_llc(cache_req_for_modifiers, IdxGlobal(params.number_of_transforms),
                                                      dimension_data.length);
      scratch_ptr_1 =
          detail::make_shared<Scalar>(2 * dimension_data.length * static_cast<std::size_t>(num_batches_in_llc), queue);
      scratch_ptr_2 =
          detail::make_shared<Scalar>(2 * dimension_data.length * static_cast<std::size_t>(num_batches_in_llc), queue);
      dimension_data.num_batches_in_l2 = num_batches_in_llc;

      if (dimension_data.is_prime) {
        // only need populate the scans and build transpose kernels
        precompute_scan_impl(dimension_data, static_cast<std::size_t>(dimension_data.num_backward_factors),
                             static_cast<std::size_t>(dimension_data.num_forward_factors),
                             dimension_data.factors_and_scan.get() + 3 * dimension_data.num_forward_factors);
      }
    } else {
      // TODO: accuractely calculate the scratch space required when there are more than one global level sizes to
      // ensure least amount of evictions
      throw internal_error("Scratch space calculation for more than one global level dimensions is not handled");
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
      bool skip_scaling = true;
      if (i == n_kernels - 1) {
        skip_scaling = false;
      }
      dimensions.emplace_back(build_w_spec_const<PORTFFT_SUBGROUP_SIZES>(i, skip_scaling));
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
      return dispatch_dimensions(in, out, in_imag, out_imag, dependencies, params.forward_strides,
                                 params.backward_strides, params.forward_distance, params.backward_distance,
                                 params.forward_offset, params.backward_offset, compute_direction);
    }
    return dispatch_dimensions(in, out, in_imag, out_imag, dependencies, params.backward_strides,
                               params.forward_strides, params.backward_distance, params.forward_distance,
                               params.backward_offset, params.forward_offset, compute_direction);
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
   * @param input_strides strides between input elements for each dimension of one FFT
   * @param output_strides strides between output elements for each dimension of one FFT
   * @param input_distance distance between the starts of input data for two consecutive FFTs
   * @param output_distance distance between the starts of output data for two consecutive FFTs
   * @param input_offset offset into input allocation where the data for FFTs start
   * @param output_offset offset into output allocation where the data for FFTs start
   * @param compute_direction direction of compute, forward / backward
   * @return sycl::event
   */
  template <typename TIn, typename TOut>
  sycl::event dispatch_dimensions(const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                                  const std::vector<sycl::event>& dependencies,
                                  const std::vector<std::size_t>& input_strides,
                                  const std::vector<std::size_t>& output_strides, std::size_t input_distance,
                                  std::size_t output_distance, std::size_t input_offset, std::size_t output_offset,
                                  direction compute_direction) {
    PORTFFT_LOG_FUNCTION_ENTRY();
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

    PORTFFT_LOG_TRACE("Dispatching the kernel for the last dimension");
    sycl::event previous_event =
        dispatch_kernel_1d(in, out, in_imag, out_imag, dependencies, params.number_of_transforms * outer_size,
                           input_stride_0, output_stride_0, input_distance, output_distance, input_offset,
                           output_offset, dimensions.back(), compute_direction);
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
            out, out, out_imag, out_imag, previous_events, inner_size, inner_size, inner_size, 1, 1,
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
   * @param input_stride stride between input elements of one FFT
   * @param output_stride stride between output elements of one FFT
   * @param input_distance distance between the starts of input data for two consecutive FFTs
   * @param output_distance distance between the starts of output data for two consecutive FFTs
   * @param input_offset offset into input allocation where the data for FFTs start
   * @param output_offset offset into output allocation where the data for FFTs start
   * @param dimension_data data for the dimension this call will work on
   * @param compute_direction direction of compute, forward / backward
   * @return sycl::event
   */
  template <typename TIn, typename TOut>
  sycl::event dispatch_kernel_1d(const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                                 const std::vector<sycl::event>& dependencies, std::size_t n_transforms,
                                 std::size_t input_stride, std::size_t output_stride, std::size_t input_distance,
                                 std::size_t output_distance, std::size_t input_offset, std::size_t output_offset,
                                 dimension_struct& dimension_data, direction compute_direction) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    return dispatch_kernel_1d_helper<TIn, TOut, PORTFFT_SUBGROUP_SIZES>(
        in, out, in_imag, out_imag, dependencies, n_transforms, input_stride, output_stride, input_distance,
        output_distance, input_offset, output_offset, dimension_data, compute_direction);
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
   * @param input_stride stride between input elements of one FFT
   * @param output_stride stride between output elements of one FFT
   * @param input_distance distance between the starts of input data for two consecutive FFTs
   * @param output_distance distance between the starts of output data for two consecutive FFTs
   * @param input_offset offset into input allocation where the data for FFTs start
   * @param output_offset offset into output allocation where the data for FFTs start
   * @param dimension_data data for the dimension this call will work on
   * @param compute_direction direction of compute, forward / backward
   * @return sycl::event
   */
  template <typename TIn, typename TOut, Idx SubgroupSize, Idx... OtherSGSizes>
  sycl::event dispatch_kernel_1d_helper(const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                                        const std::vector<sycl::event>& dependencies, std::size_t n_transforms,
                                        std::size_t input_stride, std::size_t output_stride, std::size_t input_distance,
                                        std::size_t output_distance, std::size_t input_offset,
                                        std::size_t output_offset, dimension_struct& dimension_data,
                                        direction compute_direction) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    if (SubgroupSize == dimension_data.used_sg_size) {
      const bool input_packed = input_distance == dimension_data.length && input_stride == 1;
      const bool output_packed = output_distance == dimension_data.length && output_stride == 1;
      const bool input_batch_interleaved = input_distance == 1 && input_stride == n_transforms;
      const bool output_batch_interleaved = output_distance == 1 && output_stride == n_transforms;
      for (kernel_data_struct kernel_data : dimension_data.forward_kernels) {
        std::size_t minimum_local_mem_required;
        if (input_batch_interleaved) {
          minimum_local_mem_required = num_scalars_in_local_mem<detail::layout::BATCH_INTERLEAVED>(
                                           kernel_data.level, kernel_data.length, SubgroupSize, kernel_data.factors,
                                           kernel_data.num_sgs_per_wg) *
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
      if (input_packed && output_packed) {
        return run_kernel<detail::layout::PACKED, detail::layout::PACKED, SubgroupSize>(
            in, out, in_imag, out_imag, dependencies, n_transforms, input_offset, output_offset, dimension_data,
            compute_direction);
      }
      if (input_batch_interleaved && output_packed && in != out) {
        return run_kernel<detail::layout::BATCH_INTERLEAVED, detail::layout::PACKED, SubgroupSize>(
            in, out, in_imag, out_imag, dependencies, n_transforms, input_offset, output_offset, dimension_data,
            compute_direction);
      }
      if (input_packed && output_batch_interleaved && in != out) {
        return run_kernel<detail::layout::PACKED, detail::layout::BATCH_INTERLEAVED, SubgroupSize>(
            in, out, in_imag, out_imag, dependencies, n_transforms, input_offset, output_offset, dimension_data,
            compute_direction);
      }
      if (input_batch_interleaved && output_batch_interleaved) {
        return run_kernel<detail::layout::BATCH_INTERLEAVED, detail::layout::BATCH_INTERLEAVED, SubgroupSize>(
            in, out, in_imag, out_imag, dependencies, n_transforms, input_offset, output_offset, dimension_data,
            compute_direction);
      }
      throw unsupported_configuration("Only PACKED or BATCH_INTERLEAVED transforms are supported");
    }
    if constexpr (sizeof...(OtherSGSizes) == 0) {
      throw invalid_configuration("None of the compiled subgroup sizes are supported by the device!");
    } else {
      return dispatch_kernel_1d_helper<TIn, TOut, OtherSGSizes...>(
          in, out, in_imag, out_imag, dependencies, n_transforms, input_stride, output_stride, input_distance,
          output_distance, input_offset, output_offset, dimension_data, compute_direction);
    }
  }

  /**
   * Struct for dispatching `run_kernel()` call.
   *
   * @tparam LayoutIn Input Layout
   * @tparam LayoutOut Output Layout
   * @tparam SubgroupSize size of the subgroup
   * @tparam TIn Type of the input USM pointer or buffer
   * @tparam TOut Type of the output USM pointer or buffer
   */
  template <detail::layout LayoutIn, detail::layout LayoutOut, Idx SubgroupSize, typename TIn, typename TOut>
  struct run_kernel_struct {
    // Dummy parameter is needed as only partial specializations are allowed without specializing the containing class
    template <detail::level Lev, typename Dummy>
    struct inner {
      static sycl::event execute(committed_descriptor_impl& desc, const TIn& in, TOut& out, const TIn& in_imag,
                                 TOut& out_imag, const std::vector<sycl::event>& dependencies, std::size_t n_transforms,
                                 std::size_t forward_offset, std::size_t backward_offset,
                                 dimension_struct& dimension_data, direction compute_direction);
    };
  };

  /**
   * Common interface to run the kernel called by compute_forward and compute_backward
   *
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
   * @param dimension_data data for the dimension this call will work on
   * @param compute_direction direction of fft, forward / backward
   * @return sycl::event
   */
  template <detail::layout LayoutIn, detail::layout LayoutOut, Idx SubgroupSize, typename TIn, typename TOut>
  sycl::event run_kernel(const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                         const std::vector<sycl::event>& dependencies, std::size_t n_transforms,
                         std::size_t input_offset, std::size_t output_offset, dimension_struct& dimension_data,
                         direction compute_direction) {
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
    return dispatch<run_kernel_struct<LayoutIn, LayoutOut, SubgroupSize, TInReinterpret, TOutReinterpret>>(
        dimension_data.level, detail::reinterpret<const Scalar>(in), detail::reinterpret<Scalar>(out),
        detail::reinterpret<const Scalar>(in_imag), detail::reinterpret<Scalar>(out_imag), dependencies,
        static_cast<IdxGlobal>(n_transforms), static_cast<IdxGlobal>(vec_multiplier * input_offset),
        static_cast<IdxGlobal>(vec_multiplier * output_offset), dimension_data, compute_direction);
  }
};

}  // namespace detail
}  // namespace portfft

#endif
