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
#include <common/logging.hpp>
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
 * @tparam LayoutIn Input layout
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
 * @param global_data global data for the kernel
 * @param twiddles pointer containing twiddles
 * @param scaling_factor Scaling factor applied to the result
 */
template <direction Dir, detail::layout LayoutIn, int FactorWI, int FactorSG, int SubgroupSize, typename T>
PORTFFT_INLINE void subgroup_impl(const T* input, T* output, T* loc, T* loc_twiddles, std::size_t n_transforms,
                                  global_data_struct global_data, const T* twiddles, T scaling_factor) {
  global_data.log_message_global(__func__, "entered", "FactorWI", FactorWI, "FactorSG", FactorSG, "n_transforms",
                                 n_transforms);
  constexpr int NRealsPerWI = 2 * FactorWI;
  auto loc_twiddles_view = make_padded_view<pad::DONT_PAD, 0>(loc_twiddles);

  T priv[NRealsPerWI];
  std::size_t subgroup_local_id = global_data.sg.get_local_linear_id();
  std::size_t subgroup_id = global_data.sg.get_group_id();
  std::size_t n_sgs_in_wg = global_data.it.get_local_range(0) / SubgroupSize;
  std::size_t id_of_sg_in_kernel = subgroup_id + global_data.it.get_group_linear_id() * n_sgs_in_wg;
  std::size_t n_sgs_in_kernel = global_data.it.get_group_range(0) * n_sgs_in_wg;

  std::size_t n_ffts_per_sg = SubgroupSize / FactorSG;
  std::size_t max_wis_working = n_ffts_per_sg * FactorSG;
  std::size_t n_reals_per_fft = FactorSG * NRealsPerWI;
  std::size_t n_reals_per_sg = n_ffts_per_sg * n_reals_per_fft;
  std::size_t id_of_fft_in_sg = subgroup_local_id / FactorSG;
  std::size_t id_of_wi_in_fft = subgroup_local_id % FactorSG;
  // the +1 is needed for workitems not working on useful data so they also
  // contribute to subgroup algorithms and data transfers in last iteration
  std::size_t rounded_up_n_ffts =
      round_up_to_multiple(n_transforms, n_ffts_per_sg) + (subgroup_local_id >= max_wis_working);

  std::size_t id_of_fft_in_kernel;
  std::size_t n_ffts_in_kernel;
  if constexpr (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
    id_of_fft_in_kernel = global_data.it.get_group(0) * global_data.it.get_local_range(0) / 2;
    n_ffts_in_kernel = global_data.it.get_group_range(0) * global_data.it.get_local_range(0) / 2;
  } else {
    id_of_fft_in_kernel = id_of_sg_in_kernel * n_ffts_per_sg + id_of_fft_in_sg;
    n_ffts_in_kernel = n_sgs_in_kernel * n_ffts_per_sg;
  }

  constexpr std::size_t BankLinesPerPad = 1;

  global_data.log_message_global(__func__, "loading sg twiddles from global to local memory");
  global2local<level::WORKGROUP, SubgroupSize>(global_data, twiddles, loc_twiddles_view, NRealsPerWI * FactorSG);
  sycl::group_barrier(global_data.it.get_group());
  global_data.log_dump_local("twiddles in local memory:", loc_twiddles, NRealsPerWI * FactorSG);

  for (std::size_t i = id_of_fft_in_kernel; i < rounded_up_n_ffts; i += n_ffts_in_kernel) {
    bool working = subgroup_local_id < max_wis_working && i < n_transforms;
    std::size_t n_ffts_worked_on_by_sg = sycl::min(n_transforms - (i - id_of_fft_in_sg), n_ffts_per_sg);

    if constexpr (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
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
      std::size_t id_of_fft_in_sub_batch = global_data.sg.get_group_id() * n_ffts_per_sg + id_of_fft_in_sg;
      std::size_t max_num_batches_local_mem = n_sgs_in_wg * SubgroupSize / 2;
      std::size_t num_batches_in_local_mem = [=]() {
        if (i + global_data.it.get_local_range(0) / 2 < n_transforms) {
          return global_data.it.get_local_range(0) / 2;
        }
        return n_transforms - i;
      }();
      std::size_t rounded_up_sub_batches = detail::round_up_to_multiple(num_batches_in_local_mem, n_ffts_per_sg);

      if (global_data.it.get_local_linear_id() / 2 < num_batches_in_local_mem) {
        global_data.log_message_global(__func__, "loading transposed data from global to local memory");
        // load / store in a transposed manner
        global2local_transposed<detail::level::WORKGROUP, detail::pad::DO_PAD, BankLinesPerPad, T>(
            global_data, input, loc, 2 * i, FactorWI * FactorSG, n_transforms, max_num_batches_local_mem);
      }
      sycl::group_barrier(global_data.it.get_group());
      global_data.log_dump_local("data loaded to local memory:", loc, NRealsPerWI * FactorSG);
      for (std::size_t sub_batch = id_of_fft_in_sub_batch; sub_batch < rounded_up_sub_batches;
           sub_batch += n_sgs_in_wg * n_ffts_per_sg) {
        bool working_inner = sub_batch < num_batches_in_local_mem && subgroup_local_id < max_wis_working;
        if (working_inner) {
          global_data.log_message_global(__func__, "loading transposed data from local to private memory");
          // load from local memory in a transposed manner
          local2private_transposed<FactorWI, detail::pad::DO_PAD, BankLinesPerPad>(
              global_data, loc, priv, static_cast<int>(id_of_wi_in_fft), static_cast<int>(sub_batch),
              static_cast<int>(max_num_batches_local_mem));
          global_data.log_dump_private("data loaded in registers:", priv, NRealsPerWI);
        }
        sg_dft<Dir, FactorWI, FactorSG>(priv, global_data.sg, loc_twiddles);
        if (working_inner) {
          global_data.log_dump_private("data in registers after computation:", priv, NRealsPerWI);
        }
        unrolled_loop<0, NRealsPerWI, 2>([&](int idx) PORTFFT_INLINE {
          priv[idx] *= scaling_factor;
          priv[idx + 1] *= scaling_factor;
        });
        if (working_inner) {
          global_data.log_dump_private("data in registers after scaling:", priv, NRealsPerWI);
        }
        if constexpr (SubgroupSize == FactorSG) {
          if (working_inner) {
            global_data.log_message_global(
                __func__, "storing transposed data from private to global memory (SubgroupSize == FactorSG)");
            // Store directly from registers for fully coalesced accesses
            store_transposed<NRealsPerWI, detail::pad::DONT_PAD, 0>(global_data, priv, output, id_of_wi_in_fft,
                                                                    FactorSG, (i + sub_batch) * n_reals_per_fft);
          }
        } else {
          if (working_inner) {
            global_data.log_message_global(
                __func__, "storing transposed data from private to local memory (SubgroupSize != FactorSG)");
            // Store back to local memory only
            private2local_transposed<FactorWI, detail::pad::DO_PAD, BankLinesPerPad>(
                global_data, priv, loc, static_cast<int>(id_of_wi_in_fft), FactorSG, static_cast<int>(sub_batch),
                static_cast<int>(max_num_batches_local_mem));
          }
        }
      }
      if constexpr (SubgroupSize != FactorSG) {
        global_data.log_dump_local("computed data in local memory:", loc, NRealsPerWI * FactorSG);
        global_data.log_message_global(
            __func__, "storing transposed data from local to global memory (SubgroupSize != FactorSG)");
        // store back all loaded batches at once.
        local2global_transposed<detail::pad::DO_PAD, BankLinesPerPad>(
            global_data, FactorWI * FactorSG, num_batches_in_local_mem, max_num_batches_local_mem, loc, output,
            i * n_reals_per_fft);
      }
      sycl::group_barrier(global_data.it.get_group());
    } else {
      // Codepath taken if input is not transposed
      auto local_view = make_padded_view<pad::DO_PAD, BankLinesPerPad>(loc);

      global_data.log_message_global(__func__, "loading non-transposed data from global to local memory");
      global2local<level::SUBGROUP, SubgroupSize>(
          global_data, input, local_view, n_ffts_worked_on_by_sg * n_reals_per_fft,
          n_reals_per_fft * (i - id_of_fft_in_sg), subgroup_id * n_reals_per_sg);

      sycl::group_barrier(global_data.sg);
      if (working) {
        global_data.log_message_global(__func__, "loading non-transposed data from local to private memory");
        local2private<NRealsPerWI>(global_data, local_view, priv, subgroup_local_id, NRealsPerWI,
                                   subgroup_id * n_reals_per_sg);
        global_data.log_dump_private("data loaded in registers:", priv, NRealsPerWI);
      }
      sycl::group_barrier(global_data.sg);

      sg_dft<Dir, FactorWI, FactorSG>(priv, global_data.sg, loc_twiddles);
      if (working) {
        global_data.log_dump_private("data in registers after computation:", priv, NRealsPerWI);
      }
      unrolled_loop<0, NRealsPerWI, 2>([&](int i) PORTFFT_INLINE {
        priv[i] *= scaling_factor;
        priv[i + 1] *= scaling_factor;
      });
      if (working) {
        global_data.log_dump_private("data in registers after scaling:", priv, NRealsPerWI);
      }
      if constexpr (FactorSG == SubgroupSize) {
        // in this case we get fully coalesced memory access even without going through local memory
        // TODO we may want to tune maximal `FactorSG` for which we use direct stores.
        if (working) {
          global_data.log_message_global(
              __func__, "storing transposed data from private to global memory (FactorSG == SubgroupSize)");
          store_transposed<NRealsPerWI, pad::DONT_PAD, BankLinesPerPad>(
              global_data, priv, output, id_of_wi_in_fft, FactorSG,
              i * n_reals_per_sg + id_of_fft_in_sg * n_reals_per_fft);
        }
      } else {
        if (working) {
          global_data.log_message_global(
              __func__, "storing transposed data from private to local memory (FactorSG != SubgroupSize)");
          store_transposed<NRealsPerWI, pad::DO_PAD, BankLinesPerPad>(
              global_data, priv, loc, id_of_wi_in_fft, FactorSG,
              subgroup_id * n_reals_per_sg + id_of_fft_in_sg * n_reals_per_fft);
        }
        sycl::group_barrier(global_data.sg);
        global_data.log_dump_local("computed data in local memory:", loc, NRealsPerWI * FactorSG);
        global_data.log_message_global(
            __func__, "storing transposed data from local to global memory (FactorSG != SubgroupSize)");
        local2global<level::SUBGROUP, SubgroupSize>(
            global_data, local_view, output, n_ffts_worked_on_by_sg * n_reals_per_fft, subgroup_id * n_reals_per_sg,
            n_reals_per_fft * (i - id_of_fft_in_sg));
        sycl::group_barrier(global_data.sg);
      }
    }
  }
  global_data.log_message_global(__func__, "exited");
}

/**
 * Dispatch cross sg implementation for different work-item factorizations.
 *
 * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam SubgroupSize size of the subgroup
 * @tparam LayoutIn Input layout
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
 * @param global_data global data for the kernel
 * @param twiddles pointer containing twiddles
 * @param scaling_factor Scaling factor applied to the result
 */
template <direction Dir, detail::layout LayoutIn, std::size_t SubgroupSize, typename T, typename SizeList>
PORTFFT_INLINE void subgroup_dispatch_impl(int factor_wi, int factor_sg, const T* input, T* output, T* loc,
                                           T* loc_twiddles, std::size_t n_transforms, global_data_struct global_data,
                                           const T* twiddles, T scaling_factor) {
  if constexpr (!SizeList::ListEnd) {
    constexpr int ThisSize = SizeList::Size;
    // This factorization is duplicated in the dispatch logic on the host.
    // The CT and spec constant factors should match.
    constexpr int CtFactorSg = factorize_sg(ThisSize, SubgroupSize);
    constexpr int CtFactorWi = ThisSize / CtFactorSg;
    if (factor_sg * factor_wi == ThisSize) {
      if constexpr (!fits_in_wi<T>(ThisSize) && fits_in_wi<T>(CtFactorWi) && (CtFactorSg <= SubgroupSize)) {
        detail::subgroup_impl<Dir, LayoutIn, CtFactorWi, CtFactorSg, SubgroupSize>(
            input, output, loc, loc_twiddles, n_transforms, global_data, twiddles, scaling_factor);
      }
    } else {
      subgroup_dispatch_impl<Dir, LayoutIn, SubgroupSize, T, typename SizeList::child_t>(
          factor_wi, factor_sg, input, output, loc, loc_twiddles, n_transforms, global_data, twiddles, scaling_factor);
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
    Scalar* res = sycl::aligned_alloc_device<Scalar>(
        alignof(sycl::vec<Scalar, PORTFFT_VEC_LOAD_BYTES / sizeof(Scalar)>), desc.params.lengths[0] * 2, desc.queue);
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
template <direction Dir, detail::layout LayoutIn, int SubgroupSize, typename TIn, typename TOut>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::run_kernel_struct<Dir, LayoutIn, SubgroupSize, TIn,
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
        num_scalars_in_local_mem_struct::template inner<detail::level::SUBGROUP, LayoutIn, Dummy>::execute(desc);
    std::size_t twiddle_elements = 2 * fft_size;
    return desc.queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.use_kernel_bundle(desc.exec_bundle);
      auto in_acc_or_usm = detail::get_access<const Scalar>(in, cgh);
      auto out_acc_or_usm = detail::get_access<Scalar>(out, cgh);
      sycl::local_accessor<Scalar, 1> loc(local_elements, cgh);
      sycl::local_accessor<Scalar, 1> loc_twiddles(twiddle_elements, cgh);
#ifdef PORTFFT_LOG
      sycl::stream s{1024 * 16, 1024, cgh};
#endif
      cgh.parallel_for<detail::subgroup_kernel<Scalar, Domain, Dir, Mem, LayoutIn, SubgroupSize>>(
          sycl::nd_range<1>{{global_size}, {SubgroupSize * desc.num_sgs_per_wg}}, [=
      ](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
            int factor_wi = kh.get_specialization_constant<detail::FactorWISpecConst>();
            int factor_sg = kh.get_specialization_constant<detail::FactorSGSpecConst>();
            detail::global_data_struct global_data{
#ifdef PORTFFT_LOG
                s,
#endif
                it};
            global_data.log_message_global("Running subgroup kernel");
            detail::subgroup_dispatch_impl<Dir, LayoutIn, SubgroupSize, Scalar, detail::cooley_tukey_size_list_t>(
                factor_wi, factor_sg, &in_acc_or_usm[0], &out_acc_or_usm[0], &loc[0], &loc_twiddles[0], n_transforms,
                global_data, twiddles, scale_factor);
            global_data.log_message_global("Exiting subgroup kernel");
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
template <detail::layout LayoutIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::SUBGROUP, LayoutIn,
                                                                                    Dummy> {
  static std::size_t execute(committed_descriptor& desc) {
    if constexpr (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
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
