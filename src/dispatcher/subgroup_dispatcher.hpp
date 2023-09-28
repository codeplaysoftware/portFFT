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

#include <common/cooley_tukey_compiled_sizes.hpp>
#include <common/helpers.hpp>
#include <common/subgroup.hpp>
#include <common/transfers.hpp>
#include <descriptor.hpp>
#include <enums.hpp>

namespace portfft {
namespace detail {
// specialization constants
constexpr static sycl::specialization_id<Idx> FactorWISpecConst{};
constexpr static sycl::specialization_id<Idx> FactorSGSpecConst{};

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
IdxGlobal get_global_size_subgroup(IdxGlobal n_transforms, IdxGlobal factor_sg, IdxGlobal subgroup_size,
                                     IdxGlobal num_sgs_per_wg, IdxGlobal n_compute_units) {
  IdxGlobal maximum_n_sgs = 2 * n_compute_units * 64;
  IdxGlobal maximum_n_wgs = maximum_n_sgs / num_sgs_per_wg;
  IdxGlobal wg_size = subgroup_size * num_sgs_per_wg;

  IdxGlobal n_ffts_per_wg = (subgroup_size / factor_sg) * num_sgs_per_wg;
  IdxGlobal n_wgs_we_can_utilize = divide_ceil(n_transforms, n_ffts_per_wg);
  return wg_size * sycl::min(maximum_n_wgs, n_wgs_we_can_utilize);
}

/**
 * Implementation of FFT for sizes that can be done by a subgroup.
 *
 * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam TransposeIn Whether or not the input is transposed
 * @tparam FactorWI factor of the FFT size. How many elements per FFT are processed by one workitem
 * @tparam FactorSG factor of the FFT size. How many workitems in a subgroup work on the same FFT
 * @tparam SubgroupSize size of the subgroup
 * @tparam T type of the scalar used for computations
 * @param input accessor or pointer to global memory containing input data
 * @param output accessor or pointer to global memory for output data
 * @param loc local accessor. Must have enough space for 2*FactorWI*FactorSG*SubgroupSize
 * values
 * @param loc_twiddles local accessor for twiddle factors. Must have enough space for 2*FactorWI*FactorSG
 * values
 * @param n_transforms number of FT transforms to do in one call
 * @param it sycl::nd_item<1> for the kernel launch
 * @param twiddles pointer containing twiddles
 * @param scaling_factor Scaling factor applied to the result
 */
template <direction Dir, detail::transpose TransposeIn, Idx FactorWI, Idx FactorSG, Idx SubgroupSize, typename T>
__attribute__((always_inline)) inline void subgroup_impl(const T* input, T* output, T* loc, T* loc_twiddles,
                                                         IdxGlobal n_transforms, sycl::nd_item<1> it,
                                                         const T* twiddles, T scaling_factor) {
  constexpr int NRealsPerWI = 2 * FactorWI;

  T priv[NRealsPerWI];
  sycl::sub_group sg = it.get_sub_group();
  Idx subgroup_local_id = static_cast<Idx>(sg.get_local_linear_id());
  Idx subgroup_id = static_cast<Idx>(sg.get_group_id());
  Idx n_sgs_in_wg = static_cast<Idx>(it.get_local_range(0)) / SubgroupSize;
  IdxGlobal id_of_sg_in_kernel = subgroup_id + static_cast<IdxGlobal>(it.get_group_linear_id()) * n_sgs_in_wg;
  IdxGlobal n_sgs_in_kernel = it.get_group_range(0) * n_sgs_in_wg;

  Idx n_ffts_per_sg = SubgroupSize / FactorSG;
  Idx max_wis_working = n_ffts_per_sg * FactorSG;
  Idx n_reals_per_fft = FactorSG * NRealsPerWI;
  Idx n_reals_per_sg = n_ffts_per_sg * n_reals_per_fft;
  Idx id_of_fft_in_sg = subgroup_local_id / FactorSG;
  Idx id_of_wi_in_fft = subgroup_local_id % FactorSG;
  // the +1 is needed for workitems not working on useful data so they also
  // contribute to subgroup algorithms and data transfers in last iteration
  IdxGlobal rounded_up_n_ffts =
      round_up_to_multiple(n_transforms, n_ffts_per_sg) + (subgroup_local_id >= max_wis_working);

  IdxGlobal id_of_fft_in_kernel;
  IdxGlobal n_ffts_in_kernel;
  if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
    id_of_fft_in_kernel = it.get_group(0) * it.get_local_range(0) / 2;
    n_ffts_in_kernel = it.get_group_range(0) * it.get_local_range(0) / 2;
  } else {
    id_of_fft_in_kernel = id_of_sg_in_kernel * n_ffts_per_sg + id_of_fft_in_sg;
    n_ffts_in_kernel = n_sgs_in_kernel * n_ffts_per_sg;
  }

  constexpr Idx BankLinesPerPad = 1;

  global2local<level::WORKGROUP, SubgroupSize, pad::DONT_PAD, 0>(it, twiddles, loc_twiddles, NRealsPerWI * FactorSG);
  sycl::group_barrier(it.get_group());

  for (IdxGlobal i = id_of_fft_in_kernel; i < rounded_up_n_ffts; i += n_ffts_in_kernel) {
    bool working = subgroup_local_id < max_wis_working && i < n_transforms;
    Idx n_ffts_worked_on_by_sg = sycl::min(static_cast<Idx>(n_transforms - i) + id_of_fft_in_sg, n_ffts_per_sg);

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
      Idx id_of_fft_in_sub_batch = subgroup_id * n_ffts_per_sg + id_of_fft_in_sg;
      Idx max_num_batches_local_mem = n_sgs_in_wg * SubgroupSize / 2;
      Idx num_batches_in_local_mem = [=]() {
        if (i + it.get_local_range(0) / 2 < n_transforms) {
          return it.get_local_range(0) / 2;
        }
        return n_transforms - i;
       
      }();
      Idx rounded_up_sub_batches = detail::round_up_to_multiple(num_batches_in_local_mem, n_ffts_per_sg);

      if (it.get_local_linear_id() / 2 < num_batches_in_local_mem) {
        // load / store in a transposed manner
        global2local_transposed<detail::level::WORKGROUP, detail::pad::DO_PAD, BankLinesPerPad, T>(
            it, input, loc, 2 * i, FactorWI * FactorSG, n_transforms, max_num_batches_local_mem);
      }
      sycl::group_barrier(it.get_group());
      for (Idx sub_batch = id_of_fft_in_sub_batch; sub_batch < rounded_up_sub_batches;
           sub_batch += n_sgs_in_wg * n_ffts_per_sg) {
        bool working_inner = sub_batch < num_batches_in_local_mem && subgroup_local_id < max_wis_working;
        if (working_inner) {
          // load from local memory in a transposed manner
          local2private_transposed<FactorWI, detail::pad::DO_PAD, BankLinesPerPad>(
              loc, priv, id_of_wi_in_fft, sub_batch,
              max_num_batches_local_mem);
        }
        sg_dft<Dir, FactorWI, FactorSG>(priv, sg, loc_twiddles);
        unrolled_loop<0, NRealsPerWI, 2>([&](Idx idx) __attribute__((always_inline)) {
          priv[idx] *= scaling_factor;
          priv[idx + 1] *= scaling_factor;
        });
        if constexpr (SubgroupSize == FactorSG) {
          if (working_inner) {
            // Store directly from registers for fully coalesced accesses
            store_transposed<NRealsPerWI, detail::pad::DONT_PAD, 0>(priv, output, id_of_wi_in_fft, FactorSG,
                                                                    (i + sub_batch) * n_reals_per_fft);
          }
        } else {
          if (working_inner) {
            // Store back to local memory only
            private2local_transposed<FactorWI, detail::pad::DO_PAD, BankLinesPerPad>(
                priv, loc, id_of_wi_in_fft, FactorSG, sub_batch,
                max_num_batches_local_mem);
          }
        }
      }
      if constexpr (SubgroupSize != FactorSG) {
        // store back all loaded batches at once.
        local2global_transposed<detail::pad::DO_PAD, BankLinesPerPad>(it, FactorWI * FactorSG, num_batches_in_local_mem,
                                                                      max_num_batches_local_mem, loc, output,
                                                                      i * n_reals_per_fft);
      }
      sycl::group_barrier(it.get_group());
    } else {
      // Codepath taken if input is not transposed

      global2local<level::SUBGROUP, SubgroupSize, pad::DO_PAD, BankLinesPerPad>(
          it, input, loc, n_ffts_worked_on_by_sg * n_reals_per_fft, n_reals_per_fft * (i - id_of_fft_in_sg),
          subgroup_id * n_reals_per_sg);

      sycl::group_barrier(sg);
      if (working) {
        local2private<NRealsPerWI, pad::DO_PAD, BankLinesPerPad>(loc, priv, subgroup_local_id, NRealsPerWI,
                                                                 subgroup_id * n_reals_per_sg);
      }
      sycl::group_barrier(sg);

      sg_dft<Dir, FactorWI, FactorSG>(priv, sg, loc_twiddles);
      unrolled_loop<0, NRealsPerWI, 2>([&](Idx i) __attribute__((always_inline)) {
        priv[i] *= scaling_factor;
        priv[i + 1] *= scaling_factor;
      });
      if constexpr (FactorSG == SubgroupSize) {
        // in this case we get fully coalesced memory access even without going through local memory
        // TODO we may want to tune maximal `FactorSG` for which we use direct stores.
        if (working) {
          store_transposed<NRealsPerWI, pad::DONT_PAD, BankLinesPerPad>(
              priv, output, id_of_wi_in_fft, FactorSG, i * n_reals_per_sg + id_of_fft_in_sg * n_reals_per_fft);
        }
      } else {
        if (working) {
          store_transposed<NRealsPerWI, pad::DO_PAD, BankLinesPerPad>(
              priv, loc, id_of_wi_in_fft, FactorSG, subgroup_id * n_reals_per_sg + id_of_fft_in_sg * n_reals_per_fft);
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

/**
 * Dispatch cross sg implementation for different work-item factorizations.
 *
 * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam SubgroupSize size of the subgroup
 * @tparam TransposeIn Whether or not the input is transposed
 * @tparam T type of the scalar used for computations
 * @tparam SizeList The list of sizes that will be specialized.
 * @param factor_wi factor of fft size. How many elements are processed by 1 work-item.
 * @param factor_sg cross-subgroup factor of the fft size
 * @param input accessor or pointer to global memory containing input data
 * @param output accessor or pointer to global memory for output data
 * @param loc local accessor. Must have enough space for 2*factor_wi*factor_sg*SubgroupSize
 * values
 * @param loc_twiddles local accessor for twiddle factors. Must have enough space for 2*factor_wi*factor_sg
 * values
 * @param n_transforms number of FFT transforms to do in one call
 * @param it sycl::nd_item<1> for the kernel launch
 * @param twiddles pointer containing twiddles
 * @param scaling_factor Scaling factor applied to the result
 */
template <direction Dir, detail::transpose TransposeIn, Idx SubgroupSize, typename T, typename SizeList>
__attribute__((always_inline)) void subgroup_dispatch_impl(Idx factor_wi, Idx factor_sg, const T* input, T* output,
                                                           T* loc, T* loc_twiddles, IdxGlobal n_transforms,
                                                           sycl::nd_item<1> it, const T* twiddles, T scaling_factor) {
  if constexpr (!SizeList::ListEnd) {
    constexpr Idx ThisSize = SizeList::Size;
    // This factorization is duplicated in the dispatch logic on the host.
    // The CT and spec constant factors should match.
    constexpr Idx CtFactorSg = factorize_sg(ThisSize, SubgroupSize);
    constexpr Idx CtFactorWi = ThisSize / CtFactorSg;
    if (factor_sg * factor_wi == ThisSize) {
      if constexpr (!fits_in_wi<T>(ThisSize) && fits_in_wi<T>(CtFactorWi) && (CtFactorSg <= SubgroupSize)) {
        detail::subgroup_impl<Dir, TransposeIn, CtFactorWi, CtFactorSg, SubgroupSize>(
            input, output, loc, loc_twiddles, n_transforms, it, twiddles, scaling_factor);
      }
    } else {
      subgroup_dispatch_impl<Dir, TransposeIn, SubgroupSize, T, typename SizeList::child_t>(
          factor_wi, factor_sg, input, output, loc, loc_twiddles, n_transforms, it, twiddles, scaling_factor);
    }
  }
}
}  // namespace detail

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::SUBGROUP, Dummy> {
  static Scalar* execute(committed_descriptor& desc) {
    Idx factor_wi = desc.factors[0];
    Idx factor_sg = desc.factors[1];
    Scalar* res = sycl::aligned_alloc_device<Scalar>(
        alignof(sycl::vec<Scalar, PORTFFT_VEC_LOAD_BYTES / sizeof(Scalar)>), desc.params.lengths[0] * 2, desc.queue);
    sycl::range<2> kernel_range({static_cast<std::size_t>(factor_sg), static_cast<std::size_t>(factor_wi)});
    desc.queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(kernel_range, [=](sycl::item<2> it) {
        Idx n = static_cast<Idx>(it.get_id(0));
        Idx k = static_cast<Idx>(it.get_id(1));
        sg_calc_twiddles(factor_sg, factor_wi, n, k, res);
      });
    });
    desc.queue.wait();  // waiting once here can be better than depending on the event
                        // for all future calls to compute
    return res;
  }
};

template <typename Scalar, domain Domain>
template <direction Dir, detail::transpose TransposeIn, Idx SubgroupSize, typename TIn, typename TOut>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::run_kernel_struct<Dir, TransposeIn, SubgroupSize, TIn,
                                                               TOut>::inner<detail::level::SUBGROUP, Dummy> {
  static sycl::event execute(committed_descriptor& desc, const TIn& in, TOut& out, Scalar scale_factor,
                             const std::vector<sycl::event>& dependencies) {
    constexpr detail::memory Mem = std::is_pointer<TOut>::value ? detail::memory::USM : detail::memory::BUFFER;
    std::size_t fft_size = desc.params.lengths[0];
    std::size_t n_transforms = desc.params.number_of_transforms;
    Scalar* twiddles = desc.twiddles_forward.get();
    Idx factor_sg = desc.factors[1];
    std::size_t global_size = detail::get_global_size_subgroup<Scalar>(
        n_transforms, static_cast<std::size_t>(factor_sg), SubgroupSize, desc.num_sgs_per_wg, desc.n_compute_units);
    Idx local_elements =
        num_scalars_in_local_mem_struct::template inner<detail::level::SUBGROUP, TransposeIn, Dummy>::execute(desc);
    Idx twiddle_elements = 2 * fft_size;
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
            Idx factor_wi = kh.get_specialization_constant<detail::FactorWISpecConst>();
            Idx factor_sg = kh.get_specialization_constant<detail::FactorSGSpecConst>();
            detail::subgroup_dispatch_impl<Dir, TransposeIn, SubgroupSize, Scalar, detail::cooley_tukey_size_list_t>(
                factor_wi, factor_sg, &in_acc_or_usm[0], &out_acc_or_usm[0], &loc[0], &loc_twiddles[0], n_transforms,
                it, twiddles, scale_factor);
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
    Idx dft_length = static_cast<Idx>(desc.params.lengths[0]);
    if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
      Idx twiddle_bytes = 2 * dft_length * sizeof(Scalar);
      Idx padded_fft_bytes = detail::pad_local(2 * dft_length, Idx(1)) * sizeof(Scalar);
      Idx max_batches_in_local_mem = (desc.local_memory_size - twiddle_bytes) / padded_fft_bytes;
      Idx batches_per_sg = desc.used_sg_size / 2;
      Idx num_sgs_required = std::min(Idx(PORTFFT_SGS_IN_WG),
                                              std::max(Idx(1), max_batches_in_local_mem / batches_per_sg));
      desc.num_sgs_per_wg = num_sgs_required;
      Idx num_batches_in_local_mem = desc.used_sg_size * desc.num_sgs_per_wg / 2;
      return detail::pad_local(2 * desc.params.lengths[0] * num_batches_in_local_mem, 1);
    } else {
      Idx factor_sg = desc.factors[1];
      Idx n_ffts_per_sg = desc.used_sg_size / factor_sg;
      Idx num_scalars_per_sg = detail::pad_local(2 * desc.params.lengths[0] * n_ffts_per_sg, 1);
      Idx max_n_sgs = desc.local_memory_size / sizeof(Scalar) / num_scalars_per_sg;
      desc.num_sgs_per_wg = std::min(Idx(PORTFFT_SGS_IN_WG), std::max(Idx(1), max_n_sgs));
      return num_scalars_per_sg * desc.num_sgs_per_wg;
    }
  }
};

}  // namespace portfft

#endif  // PORTFFT_DISPATCHER_SUBGROUP_DISPATCHER_HPP
