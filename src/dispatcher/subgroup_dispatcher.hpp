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

#ifndef PORTFFT_DISPATCHER_SUBGROUP_DISPATCHER_HPP
#define PORTFFT_DISPATCHER_SUBGROUP_DISPATCHER_HPP

#include <common/helpers.hpp>
#include <common/subgroup.hpp>
#include <common/transfers.hpp>
#include <descriptor.hpp>
#include <enums.hpp>

namespace portfft {
namespace detail {
// specialization constants
constexpr static sycl::specialization_id<int> FactorWISpecConst{};
constexpr static sycl::specialization_id<int> FactorSGSpecConst{};

/**
 * Calculates the global size needed for given problem.
 *
 * @tparam T type of the scalar used for computations
 * @param n_transforms number of transforms
 * @param factor_sg cross-subgroup factor of the fft size
 * @param subgroup_size size of subgroup used by the compute kernel
 * @param num_sgs_per_wg number of subgroups in a workgroup
 * @param n_compute_units number of compute units on target device
 * @return Number of elements of size T that need to fit into local memory
 */
template <typename T>
std::size_t get_global_size_subgroup(std::size_t n_transforms, std::size_t factor_sg, std::size_t subgroup_size,
                                     std::size_t num_sgs_per_wg, std::size_t n_compute_units) {
  std::size_t maximum_n_sgs = 2 * n_compute_units * 64;
  std::size_t maximum_n_wgs = maximum_n_sgs / num_sgs_per_wg;
  std::size_t wg_size = subgroup_size * num_sgs_per_wg;

  std::size_t n_ffts_per_wg = (subgroup_size / factor_sg) * num_sgs_per_wg;
  std::size_t n_wgs_we_can_utilize = divide_ceil(n_transforms, n_ffts_per_wg);
  return wg_size * sycl::min(maximum_n_wgs, n_wgs_we_can_utilize);
}

/**
 * Implementation of FFT for sizes that can be done by a subgroup.
 *
 * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam TransposeIn Whether or not the input is transposed
 * @tparam SubgroupSize size of the subgroup
 * @tparam T type of the scalar used for computations
 * @param factor_wi factor of the FFT size. How many elements per FFT are processed by one workitem
 * @param factor_sg factor of the FFT size. How many workitems in a subgroup work on the same FFT
 * @param input accessor or pointer to global memory containing input data
 * @param output accessor or pointer to global memory for output data
 * @param loc local accessor. Must have enough space for 2*factor_wi*factor_sg*SubgroupSize
 * values
 * @param loc_twiddles local accessor for twiddle factors. Must have enough space for 2*factor_wi*factor_sg
 * values
 * @param n_transforms number of FT transforms to do in one call
 * @param it sycl::nd_item<1> for the kernel launch
 * @param twiddles pointer containing twiddles
 * @param scaling_factor Scaling factor applied to the result
 */
template <direction Dir, detail::transpose TransposeIn, int SubgroupSize, typename T>
__attribute__((always_inline)) inline void subgroup_impl(std::size_t factor_wi, std::size_t factor_sg, const T* input,
                                                         T* output, T* loc, T* loc_twiddles, std::size_t n_transforms,
                                                         sycl::nd_item<1> it, const T* twiddles, T scaling_factor) {
  constexpr std::size_t MaxFactorWI = detail::MaxFftSizeWi;
  alignas(sizeof(T) * 2) T priv[2 * MaxFactorWI];
  std::size_t n_reals_per_wi = 2 * factor_wi;

  sycl::sub_group sg = it.get_sub_group();
  std::size_t subgroup_local_id = sg.get_local_linear_id();
  std::size_t subgroup_id = sg.get_group_id();
  std::size_t n_sgs_in_wg = it.get_local_range(0) / SubgroupSize;
  std::size_t id_of_sg_in_kernel = subgroup_id + it.get_group_linear_id() * n_sgs_in_wg;
  std::size_t n_sgs_in_kernel = it.get_group_range(0) * n_sgs_in_wg;

  std::size_t n_ffts_per_sg = SubgroupSize / factor_sg;
  std::size_t max_wis_working = n_ffts_per_sg * factor_sg;
  std::size_t n_reals_per_fft = factor_sg * n_reals_per_wi;
  std::size_t n_reals_per_sg = n_ffts_per_sg * n_reals_per_fft;
  std::size_t id_of_fft_in_sg = subgroup_local_id / factor_sg;
  std::size_t id_of_wi_in_fft = subgroup_local_id % factor_sg;
  // the +1 is needed for workitems not working on useful data so they also
  // contribute to subgroup algorithms and data transfers in last iteration
  std::size_t rounded_up_n_ffts =
      round_up_to_multiple(n_transforms, n_ffts_per_sg) + (subgroup_local_id >= max_wis_working);

  std::size_t id_of_fft_in_kernel;
  std::size_t n_ffts_in_kernel;
  if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
    id_of_fft_in_kernel = it.get_group(0) * it.get_local_range(0) / 2;
    n_ffts_in_kernel = it.get_group_range(0) * it.get_local_range(0) / 2;
  } else {
    id_of_fft_in_kernel = id_of_sg_in_kernel * n_ffts_per_sg + id_of_fft_in_sg;
    n_ffts_in_kernel = n_sgs_in_kernel * n_ffts_per_sg;
  }

  constexpr std::size_t BankLinesPerPad = 1;

  global2local<level::WORKGROUP, SubgroupSize, pad::DONT_PAD>(it, twiddles, loc_twiddles, n_reals_per_wi * factor_sg,
                                                              0);
  sycl::group_barrier(it.get_group());

  for (std::size_t i = id_of_fft_in_kernel; i < rounded_up_n_ffts; i += n_ffts_in_kernel) {
    bool working = subgroup_local_id < max_wis_working && i < n_transforms;
    std::size_t n_ffts_worked_on_by_sg = sycl::min(n_transforms - (i - id_of_fft_in_sg), n_ffts_per_sg);

    if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
      /**
       * Codepath taken if the input is transposed
       * The number of batches that are loaded, is equal to half of the workgroup size.
       * Each workitem is responsible for all of either the real or complex part of the batch being loaded.
       * The data in local memory is also stored in a transposed manner, so that there are no bank conflicts
       * while storing the data.
       * Thus it is loaded in a transposed manner and stored in a transposed manner to prevent data overwrites.
       * Going ahead with the assumption that output will not be stored in a transposed manner(always out of place), it
       * would need to transpose the final result in local memory and store it to global.
       */
      std::size_t id_of_fft_in_sub_batch = sg.get_group_id() * n_ffts_per_sg + id_of_fft_in_sg;
      std::size_t max_num_batches_local_mem = n_sgs_in_wg * SubgroupSize / 2;
      std::size_t num_batches_in_local_mem = [=]() {
        if (i + it.get_local_range(0) / 2 < n_transforms) {
          return it.get_local_range(0) / 2;
        }
        return n_transforms - i;
       
      }();
      std::size_t rounded_up_sub_batches = detail::round_up_to_multiple(num_batches_in_local_mem, n_ffts_per_sg);

      if (it.get_local_linear_id() / 2 < num_batches_in_local_mem) {
        // load / store in a transposed manner
        global2local_transposed<detail::level::WORKGROUP, detail::pad::DO_PAD, T>(
            it, input, loc, 2 * i, factor_wi * factor_sg, n_transforms, max_num_batches_local_mem, BankLinesPerPad);
      }
      sycl::group_barrier(it.get_group());
      for (std::size_t sub_batch = id_of_fft_in_sub_batch; sub_batch < rounded_up_sub_batches;
           sub_batch += n_sgs_in_wg * n_ffts_per_sg) {
        bool working_inner = sub_batch < num_batches_in_local_mem && subgroup_local_id < max_wis_working;
        if (working_inner) {
          // load from local memory in a transposed manner
          local2private_transposed<detail::pad::DO_PAD>(factor_wi, loc, priv, id_of_wi_in_fft, sub_batch,
                                                        max_num_batches_local_mem, BankLinesPerPad);
        }
        T wi_private_scratch[2 * wi_temps(MaxFftSizeWi)];
        sg_dft<Dir, SubgroupSize>(factor_wi, factor_sg, priv, sg, loc_twiddles, wi_private_scratch);
#pragma clang loop unroll(full)
        for (std::size_t idx{0}; idx < n_reals_per_wi; idx += 2) {
          priv[idx] *= scaling_factor;
          priv[idx + 1] *= scaling_factor;
        }
        if (SubgroupSize == factor_sg) {
          if (working_inner) {
            // Store directly from registers for fully coalesced accesses
            store_transposed<detail::pad::DONT_PAD>(n_reals_per_wi, priv, output, id_of_wi_in_fft, factor_sg, 0,
                                                    (i + sub_batch) * n_reals_per_fft);
          }
        } else {
          if (working_inner) {
            // Store back to local memory only
            private2local_transposed<detail::pad::DO_PAD>(factor_wi, priv, loc, id_of_wi_in_fft, factor_sg, sub_batch,
                                                          max_num_batches_local_mem, BankLinesPerPad);
          }
        }
      }
      if (SubgroupSize != factor_sg) {
        // store back all loaded batches at once.
        local2global_transposed<detail::pad::DO_PAD>(it, factor_wi * factor_sg, num_batches_in_local_mem,
                                                     max_num_batches_local_mem, loc, output, i * n_reals_per_fft,
                                                     BankLinesPerPad);
      }
      sycl::group_barrier(it.get_group());
    } else {
      // Codepath taken if input is not transposed

      global2local<level::SUBGROUP, SubgroupSize, pad::DO_PAD>(it, input, loc, n_ffts_worked_on_by_sg * n_reals_per_fft,
                                                               BankLinesPerPad, n_reals_per_fft * (i - id_of_fft_in_sg),
                                                               subgroup_id * n_reals_per_sg);

      sycl::group_barrier(sg);
      if (working) {
        local2private<pad::DO_PAD>(n_reals_per_wi, loc, priv, subgroup_local_id, n_reals_per_wi, BankLinesPerPad,
                                   subgroup_id * n_reals_per_sg);
      }
      sycl::group_barrier(sg);

      T wi_private_scratch[2 * wi_temps(MaxFftSizeWi)];
      sg_dft<Dir, SubgroupSize>(factor_wi, factor_sg, priv, sg, loc_twiddles, wi_private_scratch);
#pragma clang loop unroll(full)
      for (std::size_t j{0}; j < n_reals_per_wi; j += 2) {
        priv[j] *= scaling_factor;
        priv[j + 1] *= scaling_factor;
      }
      if (factor_sg == SubgroupSize) {
        // in this case we get fully coalesced memory access even without going through local memory
        // TODO we may want to tune maximal `factor_sg` for which we use direct stores.
        if (working) {
          store_transposed<pad::DONT_PAD>(n_reals_per_wi, priv, output, id_of_wi_in_fft, factor_sg, BankLinesPerPad,
                                          i * n_reals_per_sg + id_of_fft_in_sg * n_reals_per_fft);
        }
      } else {
        if (working) {
          store_transposed<pad::DO_PAD>(n_reals_per_wi, priv, loc, id_of_wi_in_fft, factor_sg, BankLinesPerPad,
                                        subgroup_id * n_reals_per_sg + id_of_fft_in_sg * n_reals_per_fft);
        }
        sycl::group_barrier(sg);
        local2global<level::SUBGROUP, SubgroupSize, pad::DO_PAD, BankLinesPerPad>(
            it, loc, output, n_ffts_worked_on_by_sg * n_reals_per_fft, subgroup_id * n_reals_per_sg,
            n_reals_per_fft * (i - id_of_fft_in_sg));
        sycl::group_barrier(sg);
      }
    }
  }
}

}  // namespace detail

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::SUBGROUP, Dummy> {
  static Scalar* execute(committed_descriptor& desc) {
    int factor_wi = desc.factors[0];
    int factor_sg = desc.factors[1];
    Scalar* res = sycl::malloc_device<Scalar>(desc.params.lengths[0] * 2, desc.queue);
    sycl::range<2> kernel_range({static_cast<std::size_t>(factor_sg), static_cast<std::size_t>(factor_wi)});
    desc.queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(kernel_range, [=](sycl::item<2> it) {
        int n = static_cast<int>(it.get_id(0));
        int k = static_cast<int>(it.get_id(1));
        sg_calc_twiddles(factor_sg, factor_wi, n, k, res);
      });
    });
    desc.queue.wait();  // waiting once here can be better than depending on the event
                        // for all future calls to compute
    return res;
  }
};

template <typename Scalar, domain Domain>
template <direction Dir, detail::transpose TransposeIn, int SubgroupSize, typename TIn, typename TOut>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::run_kernel_struct<Dir, TransposeIn, SubgroupSize, TIn,
                                                               TOut>::inner<detail::level::SUBGROUP, Dummy> {
  static sycl::event execute(committed_descriptor& desc, const TIn& in, TOut& out, Scalar scale_factor,
                             const std::vector<sycl::event>& dependencies) {
    constexpr detail::memory Mem = std::is_pointer<TOut>::value ? detail::memory::USM : detail::memory::BUFFER;
    std::size_t fft_size = desc.params.lengths[0];
    std::size_t n_transforms = desc.params.number_of_transforms;
    Scalar* twiddles = desc.twiddles_forward.get();
    int factor_sg = desc.factors[1];
    std::size_t global_size = detail::get_global_size_subgroup<Scalar>(
        n_transforms, static_cast<std::size_t>(factor_sg), SubgroupSize, desc.num_sgs_per_wg, desc.n_compute_units);
    std::size_t local_elements =
        num_scalars_in_local_mem_struct::template inner<detail::level::SUBGROUP, TransposeIn, Dummy>::execute(desc);
    std::size_t twiddle_elements = 2 * fft_size;
    return desc.queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.use_kernel_bundle(desc.exec_bundle);
      auto in_acc_or_usm = detail::get_access<const Scalar>(in, cgh);
      auto out_acc_or_usm = detail::get_access<Scalar>(out, cgh);
      sycl::local_accessor<Scalar, 1> loc(local_elements, cgh);
      sycl::local_accessor<Scalar, 1> loc_twiddles(twiddle_elements, cgh);
      cgh.parallel_for<detail::subgroup_kernel<Scalar, Domain, Dir, Mem, TransposeIn, SubgroupSize>>(
          sycl::nd_range<1>{{global_size}, {SubgroupSize * desc.num_sgs_per_wg}}, [=
      ](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
            auto factor_wi = static_cast<std::size_t>(kh.get_specialization_constant<detail::FactorWISpecConst>());
            auto factor_sg = static_cast<std::size_t>(kh.get_specialization_constant<detail::FactorSGSpecConst>());
            detail::subgroup_impl<Dir, TransposeIn, SubgroupSize>(factor_wi, factor_sg, &in_acc_or_usm[0],
                                                                  &out_acc_or_usm[0], &loc[0], &loc_twiddles[0],
                                                                  n_transforms, it, twiddles, scale_factor);
          });
    });
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::SUBGROUP, Dummy> {
  static void execute(committed_descriptor& desc, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle) {
    in_bundle.template set_specialization_constant<detail::FactorWISpecConst>(desc.factors[0]);
    in_bundle.template set_specialization_constant<detail::FactorSGSpecConst>(desc.factors[1]);
  }
};

template <typename Scalar, domain Domain>
template <detail::transpose TransposeIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::SUBGROUP,
                                                                                    TransposeIn, Dummy> {
  static std::size_t execute(committed_descriptor& desc) {
    if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
      std::size_t twiddle_bytes = 2 * desc.params.lengths[0] * sizeof(Scalar);
      std::size_t padded_fft_bytes = detail::pad_local(2 * desc.params.lengths[0], 1) * sizeof(Scalar);
      std::size_t max_batches_in_local_mem = (desc.local_memory_size - twiddle_bytes) / padded_fft_bytes;
      std::size_t batches_per_sg = static_cast<std::size_t>(desc.used_sg_size) / 2;
      std::size_t num_sgs_required = std::min(static_cast<std::size_t>(PORTFFT_SGS_IN_WG),
                                              std::max(1UL, max_batches_in_local_mem / batches_per_sg));
      desc.num_sgs_per_wg = num_sgs_required;
      std::size_t num_batches_in_local_mem = static_cast<std::size_t>(desc.used_sg_size) * desc.num_sgs_per_wg / 2;
      return detail::pad_local(2 * desc.params.lengths[0] * num_batches_in_local_mem, 1);
    } else {
      int factor_sg = desc.factors[1];
      std::size_t n_ffts_per_sg = static_cast<std::size_t>(desc.used_sg_size / factor_sg);
      std::size_t num_scalars_per_sg = detail::pad_local(2 * desc.params.lengths[0] * n_ffts_per_sg, 1);
      std::size_t max_n_sgs = desc.local_memory_size / sizeof(Scalar) / num_scalars_per_sg;
      desc.num_sgs_per_wg = std::min(static_cast<std::size_t>(PORTFFT_SGS_IN_WG), std::max(1UL, max_n_sgs));
      return num_scalars_per_sg * desc.num_sgs_per_wg;
    }
  }
};

}  // namespace portfft

#endif  // PORTFFT_DISPATCHER_SUBGROUP_DISPATCHER_HPP
