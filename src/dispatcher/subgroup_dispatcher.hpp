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
constexpr static sycl::specialization_id<int> factor_wi_spec_const{};
constexpr static sycl::specialization_id<int> factor_sg_spec_const{};

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
  std::size_t n_wgs_we_can_utilize = divideCeil(n_transforms, n_ffts_per_wg);
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
 * @param it sycl::nd_item<1> for the kernel launch
 * @param twiddles pointer containing twiddles
 * @param scaling_factor Scaling factor applied to the result
 * @param input_stride Input stride used to read the next FFT element
 * @param output_stride Input stride used to write the next FFT element
 * @param input_distance Output distance used to read the next batch
 * @param output_distance Output distance used to write the next batch
 */
template <direction Dir, detail::layout LayoutIn, int FactorWI, int FactorSG, int SubgroupSize, typename T>
__attribute__((always_inline)) inline void subgroup_impl(const T* input, T* output, T* loc, T* loc_twiddles,
                                                         std::size_t n_transforms, sycl::nd_item<1> it,
                                                         const T* twiddles, T scaling_factor, std::size_t input_stride,
                                                         std::size_t output_stride, std::size_t input_distance,
                                                         std::size_t output_distance) {
  constexpr int N_reals_per_wi = 2 * FactorWI;

  T priv[N_reals_per_wi];
  sycl::sub_group sg = it.get_sub_group();
  std::size_t subgroup_local_id = sg.get_local_linear_id();
  std::size_t subgroup_id = sg.get_group_id();
  std::size_t n_sgs_in_wg = it.get_local_range(0) / SubgroupSize;
  std::size_t id_of_sg_in_kernel = subgroup_id + it.get_group_linear_id() * n_sgs_in_wg;
  std::size_t n_sgs_in_kernel = it.get_group_range(0) * n_sgs_in_wg;

  std::size_t n_ffts_per_sg = SubgroupSize / FactorSG;
  std::size_t max_wis_working = n_ffts_per_sg * FactorSG;
  std::size_t n_reals_per_fft = FactorSG * N_reals_per_wi;
  std::size_t n_reals_per_sg = n_ffts_per_sg * n_reals_per_fft;
  std::size_t id_of_fft_in_sg = subgroup_local_id / FactorSG;
  std::size_t id_of_wi_in_fft = subgroup_local_id % FactorSG;
  // the +1 is needed for workitems not working on useful data so they also
  // contribute to subgroup algorithms and data transfers in last iteration
  std::size_t rounded_up_n_ffts =
      roundUpToMultiple(n_transforms, n_ffts_per_sg) + (subgroup_local_id >= max_wis_working);

  std::size_t id_of_fft_in_kernel;
  std::size_t n_ffts_in_kernel;
  if constexpr (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
    id_of_fft_in_kernel = it.get_group(0) * it.get_local_range(0) / 2;
    n_ffts_in_kernel = it.get_group_range(0) * it.get_local_range(0) / 2;
  } else {
    id_of_fft_in_kernel = id_of_sg_in_kernel * n_ffts_per_sg + id_of_fft_in_sg;
    n_ffts_in_kernel = n_sgs_in_kernel * n_ffts_per_sg;
  }

  global2local<pad::DONT_PAD, level::WORKGROUP, SubgroupSize>(it, twiddles, loc_twiddles, N_reals_per_wi * FactorSG);
  sycl::group_barrier(it.get_group());

  for (std::size_t i = id_of_fft_in_kernel; i < rounded_up_n_ffts; i += n_ffts_in_kernel) {
    bool working = subgroup_local_id < max_wis_working && i < n_transforms;
    std::size_t batch_id = i - id_of_fft_in_sg;
    std::size_t n_ffts_worked_on_by_sg = sycl::min(n_transforms - batch_id, n_ffts_per_sg);

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
      std::size_t id_of_fft_in_sub_batch = sg.get_group_id() * n_ffts_per_sg + id_of_fft_in_sg;
      std::size_t max_num_batches_local_mem = n_sgs_in_wg * SubgroupSize / 2;
      std::size_t num_batches_in_local_mem = [=]() {
        if (i + it.get_local_range(0) / 2 < n_transforms) {
          return it.get_local_range(0) / 2;
        } else {
          return n_transforms - i;
        }
      }();
      std::size_t rounded_up_sub_batches = detail::roundUpToMultiple(num_batches_in_local_mem, n_ffts_per_sg);

      if (it.get_local_linear_id() / 2 < num_batches_in_local_mem) {
        // load / store in a transposed manner
        global2local_transposed<detail::pad::DO_PAD, detail::level::WORKGROUP, T>(
            it, input, loc, 2 * i, FactorWI * FactorSG, n_transforms, max_num_batches_local_mem);
      }
      sycl::group_barrier(it.get_group());
      for (std::size_t sub_batch = id_of_fft_in_sub_batch; sub_batch < rounded_up_sub_batches;
           sub_batch += n_sgs_in_wg * n_ffts_per_sg) {
        bool _working = sub_batch < num_batches_in_local_mem && subgroup_local_id < max_wis_working;
        if (_working) {
          // load from local memory in a transposed manner
          local2private_transposed<FactorWI, detail::pad::DO_PAD>(loc, priv, static_cast<int>(id_of_wi_in_fft),
                                                                  static_cast<int>(sub_batch),
                                                                  static_cast<int>(max_num_batches_local_mem));
        }
        sg_dft<Dir, FactorWI, FactorSG>(priv, sg, loc_twiddles);
        unrolled_loop<0, N_reals_per_wi, 2>([&](int idx) __attribute__((always_inline)) {
          priv[idx] *= scaling_factor;
          priv[idx + 1] *= scaling_factor;
        });
        if constexpr (SubgroupSize == FactorSG) {
          if (_working) {
            // Store directly from registers for fully coalesced accesses
            store_transposed<N_reals_per_wi, detail::pad::DONT_PAD>(priv, output, id_of_wi_in_fft, FactorSG,
                                                                    (i + sub_batch) * n_reals_per_fft);
          }
        } else {
          if (_working) {
            // Store back to local memory only
            private2local_transposed<FactorWI, detail::pad::DO_PAD>(priv, loc, static_cast<int>(id_of_wi_in_fft),
                                                                    FactorSG, static_cast<int>(sub_batch),
                                                                    static_cast<int>(max_num_batches_local_mem));
          }
        }
      }
      if constexpr (SubgroupSize != FactorSG) {
        // store back all loaded batches at once.
        local2global_transposed<detail::pad::DO_PAD>(it, FactorWI * FactorSG, num_batches_in_local_mem,
                                                     max_num_batches_local_mem, loc, output, i * n_reals_per_fft);
      }
      sycl::group_barrier(it.get_group());
    } else {
      // Codepath taken if input is not transposed

      global2local<pad::DO_PAD, level::SUBGROUP, SubgroupSize>(it, input, loc, n_ffts_worked_on_by_sg, n_reals_per_fft,
                                                               input_distance * batch_id, subgroup_id * n_reals_per_sg,
                                                               input_stride, input_distance);

      sycl::group_barrier(sg);
      if (working) {
        local2private<N_reals_per_wi, pad::DO_PAD>(loc, priv, subgroup_local_id, N_reals_per_wi,
                                                   subgroup_id * n_reals_per_sg);
      }
      sycl::group_barrier(sg);

      sg_dft<Dir, FactorWI, FactorSG>(priv, sg, loc_twiddles);
      unrolled_loop<0, N_reals_per_wi, 2>([&](int i) __attribute__((always_inline)) {
        priv[i] *= scaling_factor;
        priv[i + 1] *= scaling_factor;
      });
      if constexpr (FactorSG == SubgroupSize && LayoutIn == detail::layout::PACKED) {
        // in this case we get fully coalesced memory access even without going through local memory
        // TODO we may want to tune maximal `FactorSG` for which we use direct stores.
        if (working) {
          store_transposed<N_reals_per_wi, pad::DONT_PAD>(priv, output, id_of_wi_in_fft, FactorSG,
                                                          i * n_reals_per_sg + id_of_fft_in_sg * n_reals_per_fft);
        }
      } else {
        if (working) {
          store_transposed<N_reals_per_wi, pad::DO_PAD>(
              priv, loc, id_of_wi_in_fft, FactorSG, subgroup_id * n_reals_per_sg + id_of_fft_in_sg * n_reals_per_fft);
        }
        sycl::group_barrier(sg);

        local2global<pad::DO_PAD, level::SUBGROUP, SubgroupSize>(
            it, loc, output, n_ffts_worked_on_by_sg, n_reals_per_fft, subgroup_id * n_reals_per_sg,
            output_distance * batch_id, output_stride, output_distance);
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
 * @param it sycl::nd_item<1> for the kernel launch
 * @param twiddles pointer containing twiddles
 * @param scaling_factor Scaling factor applied to the result
 * @param input_stride Input stride used to read the next FFT element
 * @param output_stride Input stride used to write the next FFT element
 * @param input_distance Output distance used to read the next batch
 * @param output_distance Output distance used to write the next batch
 */
template <direction Dir, detail::layout LayoutIn, std::size_t SubgroupSize, typename T, typename SizeList>
__attribute__((always_inline)) void subgroup_dispatch_impl(int factor_wi, int factor_sg, const T* input, T* output,
                                                           T* loc, T* loc_twiddles, std::size_t n_transforms,
                                                           sycl::nd_item<1> it, const T* twiddles, T scaling_factor,
                                                           std::size_t input_stride, std::size_t output_stride,
                                                           std::size_t input_distance, std::size_t output_distance) {
  if constexpr (!SizeList::list_end) {
    constexpr int this_size = SizeList::size;
    // This factorization is duplicated in the dispatch logic on the host.
    // The CT and spec constant factors should match.
    constexpr int ct_factor_sg = factorize_sg(this_size, SubgroupSize);
    constexpr int ct_factor_wi = this_size / ct_factor_sg;
    if (factor_sg * factor_wi == this_size) {
      if constexpr (!fits_in_wi<T>(this_size) && fits_in_wi<T>(ct_factor_wi) && (ct_factor_sg <= SubgroupSize)) {
        detail::subgroup_impl<Dir, LayoutIn, ct_factor_wi, ct_factor_sg, SubgroupSize>(
            input, output, loc, loc_twiddles, n_transforms, it, twiddles, scaling_factor, input_stride, output_stride,
            input_distance, output_distance);
      }
    } else {
      subgroup_dispatch_impl<Dir, LayoutIn, SubgroupSize, T, typename SizeList::child_t>(
          factor_wi, factor_sg, input, output, loc, loc_twiddles, n_transforms, it, twiddles, scaling_factor,
          input_stride, output_stride, input_distance, output_distance);
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
template <direction Dir, detail::layout LayoutIn, int SubgroupSize, typename T_in, typename T_out>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::run_kernel_struct<Dir, LayoutIn, SubgroupSize, T_in,
                                                               T_out>::inner<detail::level::SUBGROUP, Dummy> {
  static sycl::event execute(committed_descriptor& desc, const T_in& in, T_out& out,
                             const std::vector<sycl::event>& dependencies) {
    constexpr detail::memory mem = std::is_pointer<T_out>::value ? detail::memory::USM : detail::memory::BUFFER;
    std::size_t fft_size = desc.params.lengths[0];
    std::size_t n_transforms = desc.params.number_of_transforms;
    Scalar* twiddles = desc.twiddles_forward.get();
    int factor_sg = desc.factors[1];
    std::size_t global_size = detail::get_global_size_subgroup<Scalar>(
        n_transforms, static_cast<std::size_t>(factor_sg), SubgroupSize, desc.num_sgs_per_wg, desc.n_compute_units);
    std::size_t local_elements =
        num_scalars_in_local_mem_struct::template inner<detail::level::SUBGROUP, Dummy>::execute(desc, Dir);
    std::size_t twiddle_elements = 2 * fft_size;
    auto input_strides = desc.params.get_strides(Dir);
    auto output_strides = desc.params.get_strides(inv(Dir));
    static constexpr std::size_t ElemSize = 2;
    std::size_t input_offset = input_strides[0] * ElemSize;
    std::size_t output_offset = output_strides[0] * ElemSize;
    std::size_t input_1d_stride = input_strides.back() * ElemSize;
    std::size_t output_1d_stride = output_strides.back() * ElemSize;
    auto input_distance = desc.params.get_distance(Dir) * ElemSize;
    auto output_distance = desc.params.get_distance(inv(Dir)) * ElemSize;
    auto scale_factor = desc.params.get_scale(Dir);
    sycl::event event = desc.queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.use_kernel_bundle(desc.exec_bundle.value());
      auto in_acc_or_usm = detail::get_access<const Scalar>(in, cgh);
      auto out_acc_or_usm = detail::get_access<Scalar>(out, cgh);
      sycl::local_accessor<Scalar, 1> loc(local_elements, cgh);
      sycl::local_accessor<Scalar, 1> loc_twiddles(twiddle_elements, cgh);
      cgh.parallel_for<detail::subgroup_kernel<Scalar, Domain, Dir, mem, LayoutIn, SubgroupSize>>(
          sycl::nd_range<1>{{global_size}, {SubgroupSize * desc.num_sgs_per_wg}}, [=
      ](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
            int factor_wi = kh.get_specialization_constant<detail::factor_wi_spec_const>();
            int factor_sg = kh.get_specialization_constant<detail::factor_sg_spec_const>();
            detail::subgroup_dispatch_impl<Dir, LayoutIn, SubgroupSize, Scalar, detail::cooley_tukey_size_list_t>(
                factor_wi, factor_sg, &in_acc_or_usm[0] + input_offset, &out_acc_or_usm[0] + output_offset, &loc[0],
                &loc_twiddles[0], n_transforms, it, twiddles, scale_factor, input_1d_stride, output_1d_stride,
                input_distance, output_distance);
          });
    });
    return event;
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::SUBGROUP, Dummy> {
  static void execute(committed_descriptor& desc, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle) {
    in_bundle.template set_specialization_constant<detail::factor_wi_spec_const>(desc.factors[0]);
    in_bundle.template set_specialization_constant<detail::factor_sg_spec_const>(desc.factors[1]);
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::SUBGROUP, Dummy> {
  static std::size_t execute(committed_descriptor& desc, direction dir) {
    auto input_layout = detail::get_layout(desc.params, dir);
    auto output_layout = detail::get_layout(desc.params, inv(dir));
    if (input_layout == detail::layout::BATCH_INTERLEAVED && output_layout == detail::layout::PACKED) {
      // Input is transposed and not the output
      std::size_t twiddle_bytes = 2 * desc.params.lengths[0] * sizeof(Scalar);
      std::size_t padded_fft_bytes = detail::pad_local(2 * desc.params.lengths[0]) * sizeof(Scalar);
      std::size_t max_batches_in_local_mem = (desc.local_memory_size - twiddle_bytes) / padded_fft_bytes;
      std::size_t batches_per_sg = static_cast<std::size_t>(desc.used_sg_size) / 2;
      std::size_t num_sgs_required = std::min(static_cast<std::size_t>(PORTFFT_SGS_IN_WG),
                                              std::max(1ul, max_batches_in_local_mem / batches_per_sg));
      desc.num_sgs_per_wg = num_sgs_required;
      std::size_t num_batches_in_local_mem = static_cast<std::size_t>(desc.used_sg_size) * desc.num_sgs_per_wg / 2;
      return detail::pad_local(2 * desc.params.lengths[0] * num_batches_in_local_mem);
    } else {
      int factor_sg = desc.factors[1];
      std::size_t n_ffts_per_sg = static_cast<std::size_t>(desc.used_sg_size / factor_sg);
      std::size_t num_scalars_per_sg = detail::pad_local(2 * desc.params.lengths[0] * n_ffts_per_sg);
      std::size_t max_n_sgs = desc.local_memory_size / sizeof(Scalar) / num_scalars_per_sg;
      desc.num_sgs_per_wg = std::min(static_cast<std::size_t>(PORTFFT_SGS_IN_WG), std::max(1ul, max_n_sgs));
      return num_scalars_per_sg * desc.num_sgs_per_wg;
    }
  }
};

}  // namespace portfft

#endif  // PORTFFT_DISPATCHER_SUBGROUP_DISPATCHER_HPP
