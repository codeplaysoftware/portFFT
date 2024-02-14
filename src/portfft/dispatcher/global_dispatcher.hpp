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

#ifndef PORTFFT_DISPATCHER_GLOBAL_DISPATCHER_HPP
#define PORTFFT_DISPATCHER_GLOBAL_DISPATCHER_HPP

#include <sycl/sycl.hpp>

#include <cstring>

#include "portfft/common/bluestein.hpp"
#include "portfft/common/global.hpp"
#include "portfft/common/subgroup.hpp"
#include "portfft/defines.hpp"
#include "portfft/enums.hpp"
#include "portfft/specialization_constant.hpp"

namespace portfft {
namespace detail {

/**
 * Helper function to obtain the global and local range for kernel corresponding to the factor
 * @param fft_size length of the factor
 * @param num_batches number of corresposing batches
 * @param level The implementation for the factor
 * @param n_compute_units compute_units available
 * @param subgroup_size Subgroup size chosen
 * @param n_sgs_in_wg Number of subgroups in a workgroup.
 * @return std::pair containing global and local range
 */
inline std::pair<IdxGlobal, IdxGlobal> get_launch_params(IdxGlobal fft_size, IdxGlobal num_batches, detail::level level,
                                                         Idx n_compute_units, Idx subgroup_size, Idx n_sgs_in_wg) {
  PORTFFT_LOG_FUNCTION_ENTRY();
  IdxGlobal n_available_sgs = 8 * n_compute_units * 64;
  IdxGlobal wg_size = n_sgs_in_wg * subgroup_size;
  if (level == detail::level::WORKITEM) {
    IdxGlobal n_ffts_per_wg = wg_size;
    IdxGlobal n_wgs_required = divide_ceil(num_batches, n_ffts_per_wg);
    return std::make_pair(std::min(n_wgs_required * wg_size, n_available_sgs), wg_size);
  }
  if (level == detail::level::SUBGROUP) {
    IdxGlobal n_ffts_per_sg = static_cast<IdxGlobal>(subgroup_size) / detail::factorize_sg(fft_size, subgroup_size);
    IdxGlobal n_ffts_per_wg = n_ffts_per_sg * n_sgs_in_wg;
    IdxGlobal n_wgs_required = divide_ceil(num_batches, n_ffts_per_wg);
    return std::make_pair(std::min(n_wgs_required * wg_size, n_available_sgs), wg_size);
  }
  if (level == detail::level::WORKGROUP) {
    return std::make_pair(std::min(num_batches * wg_size, n_available_sgs), wg_size);
  }
  throw internal_error("illegal level encountered");
}

/**
 * Transposes A into B, for complex inputs only
 * @param a Input pointer
 * @param b Output pointer
 * @param lda leading dimension of `a`
 * @param ldb leading dimension of `b`
 * @param num_elements Total number of complex values in the matrix
 */
template <typename T>
void complex_transpose(const T* a, T* b, IdxGlobal lda, IdxGlobal ldb, IdxGlobal num_elements) {
  PORTFFT_LOG_FUNCTION_ENTRY();
  for (IdxGlobal i = 0; i < num_elements; i++) {
    IdxGlobal j = i / ldb;
    IdxGlobal k = i % ldb;
    b[2 * i] = a[2 * k * lda + 2 * j];
    b[2 * i + 1] = a[2 * k * lda + 2 * j + 1];
  }
}

/**
 * Copied data from source  to the destination.
 * @tparam TIn Input Type
 * @tparam TOut Output Type
 * @param src sycl::buffer / pointer containing the input data. In the case SPLIT_COMPLEX storage, it contains only the
 * real part
 * @param src_imag sycl::buffer / pointer containing the imaginary part of the input in the case where storage is
 * SPLIT_COMPLEX
 * @param dst sycl::buffer / pointer containing the output data. In the case SPLIT_COMPLEX storage, it contains only the
 * real part
 * @param dst_imag sycl::buffer / pointer containing the imaginary part of the output in the case where storage is
 * SPLIT_COMPLEX
 * @param num_elements_to_copy number of elements to copy
 * @param src_stride distance between two consecutive batches in the input
 * @param dst_stride disance between two consecutive batches of the output
 * @param num_copies number of batches to copy
 * @param input_offset offset applied to the input
 * @param output_offset offset applied to the output
 * @param storage complex storage scheme: split_complex / complex_interleaved
 * @param queue sycl queue associated with the commit
 * @return
 */
template <typename TIn, typename TOut>
sycl::event trigger_device_copy(const TIn src, const TIn src_imag, TOut dst, TOut dst_imag,
                                std::size_t num_elements_to_copy, std::size_t src_stride, std::size_t dst_stride,
                                std::size_t num_copies, std::size_t input_offset, std::size_t output_offset,
                                complex_storage storage, sycl::queue& queue) {
  std::vector<sycl::event> events;
  auto trigger_device_copy_impl = [&](const TIn& input, TOut& output) {
    queue.submit([&](sycl::handler& cgh) {
      auto in_acc_or_usm = get_access(input, cgh);
      auto out_acc_or_usm = get_access(output, cgh);
      cgh.host_task([&]() {
        for (std::size_t i = 0; i < num_copies; i++) {
          events.push_back(queue.copy(&in_acc_or_usm[0] + i * src_stride + input_offset,
                                      &out_acc_or_usm[0] + i * dst_stride + output_offset, num_elements_to_copy));
        }
      });
    });
  };
  trigger_device_copy_impl(src, dst);
  if (storage == complex_storage::SPLIT_COMPLEX) {
    trigger_device_copy_impl(src_imag, dst_imag);
  }
  return queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(events);
    cgh.host_task([]() {});
  });
}

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::GLOBAL, Dummy> {
  static Scalar* execute(committed_descriptor_impl& desc, dimension_struct& dimension_data,
                         std::vector<kernel_data_struct>& kernels) {
    using idxglobal_vec_t = std::vector<IdxGlobal>;
    PORTFFT_LOG_FUNCTION_ENTRY();
    /**
     * Helper Lambda to calculate twiddles
     */
    auto calculate_twiddles = [](IdxGlobal N, IdxGlobal M, IdxGlobal& offset, Scalar* ptr) {
      for (IdxGlobal i = 0; i < N; i++) {
        for (IdxGlobal j = 0; j < M; j++) {
          double theta = -2 * M_PI * static_cast<double>(i * j) / static_cast<double>(N * M);
          ptr[offset++] = static_cast<Scalar>(std::cos(theta));
          ptr[offset++] = static_cast<Scalar>(std::sin(theta));
        }
      }
    };

    auto calculate_level_specific_twiddles = [&calculate_twiddles](Scalar* host_twiddles_ptr, Scalar* scratch_ptr,
                                                                   const kernel_data_struct& kernel_data,
                                                                   IdxGlobal& ptr_offset) {
      if (kernel_data.level == detail::level::SUBGROUP) {
        for (Idx i = 0; i < kernel_data.factors.at(0); i++) {
          for (Idx j = 0; j < kernel_data.factors.at(1); j++) {
            double theta = -2 * M_PI * static_cast<double>(i * j) /
                           static_cast<double>(kernel_data.factors.at(0) * kernel_data.factors.at(1));
            auto twiddle =
                std::complex<Scalar>(static_cast<Scalar>(std::cos(theta)), static_cast<Scalar>(std::sin(theta)));
            host_twiddles_ptr[static_cast<std::size_t>(
                ptr_offset + static_cast<IdxGlobal>(j * kernel_data.factors.at(0) + i))] = twiddle.real();
            host_twiddles_ptr[static_cast<std::size_t>(
                ptr_offset + static_cast<IdxGlobal>((j + kernel_data.factors.at(1)) * kernel_data.factors.at(0) + i))] =
                twiddle.imag();
          }
        }
        ptr_offset += static_cast<IdxGlobal>(2 * kernel_data.length);
      } else if (kernel_data.level == detail::level::WORKGROUP) {
        Idx factor_n = kernel_data.factors.at(0) * kernel_data.factors.at(1);
        Idx factor_m = kernel_data.factors.at(2) * kernel_data.factors.at(3);
        calculate_twiddles(static_cast<IdxGlobal>(kernel_data.factors.at(0)),
                           static_cast<IdxGlobal>(kernel_data.factors.at(1)), ptr_offset, host_twiddles_ptr);
        calculate_twiddles(static_cast<IdxGlobal>(kernel_data.factors.at(2)),
                           static_cast<IdxGlobal>(kernel_data.factors.at(3)), ptr_offset, host_twiddles_ptr);
        // Calculate wg twiddles and transpose them
        calculate_twiddles(static_cast<IdxGlobal>(factor_n), static_cast<IdxGlobal>(factor_m), ptr_offset,
                           host_twiddles_ptr);
        for (Idx j = 0; j < factor_n; j++) {
          detail::complex_transpose(host_twiddles_ptr + ptr_offset + 2 * j * factor_n, scratch_ptr, factor_m, factor_n,
                                    factor_n * factor_m);
          std::memcpy(host_twiddles_ptr + ptr_offset + 2 * j * factor_n, scratch_ptr, 2 * kernel_data.length);
        }
      }
    };

    auto get_sub_batches_and_factors = [&dimension_data, &kernels]() -> std::tuple<idxglobal_vec_t, idxglobal_vec_t> {
      auto get_sub_batches_and_factors_impl = [&kernels](idxglobal_vec_t& factors, idxglobal_vec_t& sub_batches,
                                                         std::size_t num_factors, std::size_t offset) -> void {
        for (std::size_t i = 0; i < num_factors; i++) {
          factors.push_back(static_cast<IdxGlobal>(kernels.at(offset + i).length));
        }
        for (std::size_t i = 0; i < num_factors; i++) {
          sub_batches.push_back(std::accumulate(factors.begin() + static_cast<long>(offset + i + 1), factors.end(),
                                                IdxGlobal(1), std::multiplies<IdxGlobal>()));
        }
        sub_batches.push_back(factors.at(factors.size() - 2));
      };
      idxglobal_vec_t factors;
      idxglobal_vec_t sub_batches;
      get_sub_batches_and_factors_impl(factors, sub_batches,
                                       static_cast<std::size_t>(dimension_data.num_forward_factors), 0);
      if (dimension_data.is_prime) {
        get_sub_batches_and_factors_impl(factors, sub_batches,
                                         static_cast<std::size_t>(dimension_data.num_backward_factors),
                                         static_cast<std::size_t>(dimension_data.num_forward_factors));
      }
      return {factors, sub_batches};
    };

    auto get_total_mem_for_twiddles = [&dimension_data, &kernels](const idxglobal_vec_t& factors,
                                                                  const idxglobal_vec_t& sub_batches) -> std::size_t {
      auto get_total_mem_for_twiddles_impl = [&kernels](const idxglobal_vec_t& factors,
                                                        const idxglobal_vec_t& sub_batches, std::size_t offset,
                                                        std::size_t num_factors) -> std::size_t {
        IdxGlobal mem_required_for_twiddles = 0;
        // account for memory required for store modifiers
        for (std::size_t i = 0; i < num_factors - 1; i++) {
          mem_required_for_twiddles += 2 * factors.at(offset + i) * sub_batches.at(offset + i);
        }
        // account for memory required for factor specific twiddles
        for (std::size_t i = 0; i < num_factors; i++) {
          const auto& kd_struct = kernels.at(offset + i);
          if (kd_struct.level == detail::level::SUBGROUP) {
            mem_required_for_twiddles += static_cast<IdxGlobal>(2 * kd_struct.length);
          }
          if (kd_struct.level == detail::level::WORKGROUP) {
            mem_required_for_twiddles +=
                static_cast<IdxGlobal>(2 * kd_struct.length) +
                static_cast<IdxGlobal>(
                    2 * std::accumulate(kd_struct.factors.begin(), kd_struct.factors.end(), 0, std::plus<Idx>()));
          }
        }
        return static_cast<std::size_t>(mem_required_for_twiddles);
      };
      std::size_t mem_required_for_twiddles = get_total_mem_for_twiddles_impl(
          factors, sub_batches, 0, static_cast<std::size_t>(dimension_data.num_forward_factors));
      if (dimension_data.is_prime) {
        mem_required_for_twiddles += get_total_mem_for_twiddles_impl(
            factors, sub_batches, static_cast<std::size_t>(dimension_data.num_forward_factors),
            static_cast<std::size_t>(dimension_data.num_backward_factors));
        // load / store modifiers for bluestein
        mem_required_for_twiddles += 4 * dimension_data.length;
      }
      return mem_required_for_twiddles;
    };

    auto get_local_memory_usage = [&desc](detail::layout layout, const kernel_data_struct& kernel_data,
                                          Idx& num_sgs_in_wg) -> std::size_t {
      if (kernel_data.level == detail::level::WORKITEM && layout == detail::layout::BATCH_INTERLEAVED) {
        return 0;
      }
      return desc.num_scalars_in_local_mem(kernel_data.level, kernel_data.length, kernel_data.used_sg_size,
                                           kernel_data.factors, num_sgs_in_wg, layout);
    };

    auto calculate_twiddles_and_populate_metadata = [&dimension_data, &kernels, &desc, &calculate_twiddles,
                                                     &calculate_level_specific_twiddles, &get_local_memory_usage](
                                                        Scalar* host_twiddles_ptr, const idxglobal_vec_t& factors,
                                                        const idxglobal_vec_t& sub_batches) -> void {
      auto calculate_twiddles_and_populate_metadata_impl =
          [&](Scalar* scratch_ptr, std::size_t num_factors, std::size_t kd_offset, IdxGlobal& ptr_offset) -> IdxGlobal {
        // calculate twiddles between factors
        IdxGlobal impl_twiddles_offset;
        for (std::size_t i = 0; i < num_factors - 1; i++) {
          calculate_twiddles(sub_batches.at(kd_offset + i), factors.at(kd_offset + i), ptr_offset, host_twiddles_ptr);
        }
        impl_twiddles_offset = ptr_offset;
        for (std::size_t i = 0; i < num_factors; i++) {
          auto& kernel_data = kernels.at(kd_offset + i);
          kernel_data.batch_size = sub_batches.at(kd_offset + i);
          Idx num_sgs_in_wg = PORTFFT_SGS_IN_WG;
          detail::layout layout = i == num_factors - 1 ? detail::layout::PACKED : detail::layout::BATCH_INTERLEAVED;
          kernel_data.local_mem_required = get_local_memory_usage(layout, kernel_data, num_sgs_in_wg);
          kernel_data.num_sgs_per_wg = num_sgs_in_wg;
          const auto [global_range, local_range] =
              get_launch_params(factors.at(kd_offset + i), sub_batches.at(kd_offset + i), kernel_data.level,
                                desc.n_compute_units, kernel_data.used_sg_size, num_sgs_in_wg);
          kernel_data.global_range = global_range;
          kernel_data.local_range = local_range;
          calculate_level_specific_twiddles(host_twiddles_ptr, scratch_ptr, kernel_data, ptr_offset);
        }
        return impl_twiddles_offset;
      };
      std::vector<Scalar> scratch_space(2 * dimension_data.length);
      IdxGlobal offset = 0;
      dimension_data.forward_impl_twiddle_offset = calculate_twiddles_and_populate_metadata_impl(
          host_twiddles_ptr, static_cast<std::size_t>(dimension_data.num_forward_factors), 0, offset);
      if (dimension_data.is_prime) {
        dimension_data.backward_twiddles_offset = offset;
        dimension_data.backward_impl_twiddle_offset = calculate_twiddles_and_populate_metadata_impl(
            host_twiddles_ptr, static_cast<std::size_t>(dimension_data.num_backward_factors),
            static_cast<std::size_t>(dimension_data.num_forward_factors), offset);
        dimension_data.bluestein_modifiers_offset = offset;
        detail::populate_bluestein_input_modifiers(host_twiddles_ptr + offset, dimension_data.committed_length,
                                                   dimension_data.length);
        offset += static_cast<IdxGlobal>(2 * dimension_data.length);
        detail::get_fft_chirp_signal(host_twiddles_ptr + offset, dimension_data.committed_length,
                                     dimension_data.length);
      }
    };

    const auto [factors, sub_batches] = get_sub_batches_and_factors();
    std::size_t mem_required_for_twiddles = get_total_mem_for_twiddles(factors, sub_batches);
    Scalar* device_twiddles_ptr = sycl::malloc_device<Scalar>(mem_required_for_twiddles, desc.queue);
    if (!device_twiddles_ptr) {
      throw internal_error("Could not allocate usm memory of size: ", mem_required_for_twiddles * sizeof(Scalar),
                           " bytes");
    }
    std::vector<Scalar> host_memory_twiddles(mem_required_for_twiddles);
    calculate_twiddles_and_populate_metadata(host_memory_twiddles.data(), factors, sub_batches);
    desc.queue.copy(host_memory_twiddles.data(), device_twiddles_ptr, mem_required_for_twiddles).wait();
    return device_twiddles_ptr;
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::GLOBAL, Dummy> {
  static void execute(committed_descriptor_impl& /*desc*/, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle,
                      Idx length, const std::vector<Idx>& factors, detail::level level, Idx factor_num,
                      Idx num_factors) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    PORTFFT_LOG_TRACE("GlobalSubImplSpecConst:", level);
    in_bundle.template set_specialization_constant<detail::GlobalSubImplSpecConst>(level);
    PORTFFT_LOG_TRACE("GlobalSpecConstNumFactors:", num_factors);
    in_bundle.template set_specialization_constant<detail::GlobalSpecConstNumFactors>(num_factors);
    PORTFFT_LOG_TRACE("GlobalSpecConstLevelNum:", factor_num);
    in_bundle.template set_specialization_constant<detail::GlobalSpecConstLevelNum>(factor_num);
    if (level == detail::level::WORKITEM || level == detail::level::WORKGROUP) {
      PORTFFT_LOG_TRACE("SpecConstFftSize:", length);
      in_bundle.template set_specialization_constant<detail::SpecConstFftSize>(length);
    } else if (level == detail::level::SUBGROUP) {
      PORTFFT_LOG_TRACE("SubgroupFactorWISpecConst:", factors[1]);
      in_bundle.template set_specialization_constant<detail::SubgroupFactorWISpecConst>(factors[1]);
      PORTFFT_LOG_TRACE("SubgroupFactorSGSpecConst:", factors[0]);
      in_bundle.template set_specialization_constant<detail::SubgroupFactorSGSpecConst>(factors[0]);
    }
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::GLOBAL, Dummy> {
  static std::size_t execute(committed_descriptor_impl& /*desc*/, std::size_t /*length*/, Idx /*used_sg_size*/,
                             const std::vector<Idx>& /*factors*/, Idx& /*num_sgs_per_wg*/, layout /*input_layout*/) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    // No work required as all work done in calculate_twiddles;
    return 0;
  }
};

template <typename Scalar, domain Domain>
template <Idx SubgroupSize, typename TIn, typename TOut>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::run_kernel_struct<SubgroupSize, TIn,
                                                                    TOut>::inner<detail::level::GLOBAL, Dummy> {
  static sycl::event execute(committed_descriptor_impl& desc, const TIn& in, TOut& out, const TIn& in_imag,
                             TOut& out_imag, const std::vector<sycl::event>& dependencies, IdxGlobal n_transforms,
                             IdxGlobal input_offset, IdxGlobal output_offset, dimension_struct& dimension_data,
                             direction compute_direction, layout /*input_layout*/) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    complex_storage storage = desc.params.complex_storage;
    const std::size_t vec_size = storage == complex_storage::INTERLEAVED_COMPLEX ? 2 : 1;
    const auto& kernels =
        compute_direction == direction::FORWARD ? dimension_data.forward_kernels : dimension_data.backward_kernels;
    const Scalar* twiddles_ptr = static_cast<const Scalar*>(kernels.at(0).twiddles_forward.get());
    std::size_t num_batches = static_cast<std::size_t>(n_transforms);
    std::size_t max_batches_in_l2 = static_cast<std::size_t>(dimension_data.num_batches_in_l2);
    std::size_t imag_offset = dimension_data.length * max_batches_in_l2;

    sycl::event event = desc.queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.host_task([&]() {});
    });

    for (std::size_t i = 0; i < num_batches; i += max_batches_in_l2) {
      if (dimension_data.is_prime) {
        std::size_t num_copies =
            i + max_batches_in_l2 < num_batches ? max_batches_in_l2 : num_batches - max_batches_in_l2;
        // TODO: look into other library implementations to check whether is it possible at all to avoid this explicit
        // copy.
        trigger_device_copy(in, in_imag, desc.scratch_ptr_1.get(), desc.scratch_ptr_1.get() + imag_offset,
                            vec_size * dimension_data.committed_length, vec_size * dimension_data.committed_length,
                            vec_size * dimension_data.length,
                            vec_size * i * dimension_data.committed_length + static_cast<std::size_t>(input_offset), 0,
                            num_copies, storage, desc.queue)
            .wait();

        detail::global_impl_driver<SubgroupSize>(
            static_cast<const Scalar*>(desc.scratch_ptr_1.get()),
            static_cast<const Scalar*>(desc.scratch_ptr_1.get() + imag_offset), desc.scratch_ptr_2.get(),
            desc.scratch_ptr_2.get() + imag_offset, desc, dimension_data, kernels, dimension_data.transpose_kernels,
            dimension_data.num_forward_factors, 0, dimension_data.forward_impl_twiddle_offset, 0, i,
            static_cast<IdxGlobal>(num_batches), 0, 0, storage, detail::elementwise_multiply::APPLIED,
            twiddles_ptr + dimension_data.bluestein_modifiers_offset + 2 * dimension_data.length)
            .wait();
        std::swap(desc.scratch_ptr_1, desc.scratch_ptr_2);
        detail::global_impl_driver<SubgroupSize>(
            static_cast<const Scalar*>(desc.scratch_ptr_1.get()),
            static_cast<const Scalar*>(desc.scratch_ptr_1.get() + imag_offset), desc.scratch_ptr_2.get(),
            desc.scratch_ptr_2.get() + imag_offset, desc, dimension_data, kernels, dimension_data.transpose_kernels,
            dimension_data.num_backward_factors, dimension_data.backward_twiddles_offset,
            dimension_data.backward_impl_twiddle_offset, std::size_t(dimension_data.num_forward_factors), i,
            static_cast<IdxGlobal>(num_batches), 0, 0, storage, detail::elementwise_multiply::NOT_APPLIED,
            twiddles_ptr + dimension_data.bluestein_modifiers_offset);

        trigger_device_copy(desc.scratch_ptr_2.get(), desc.scratch_ptr_2.get() + imag_offset, out, out_imag,
                            vec_size * dimension_data.committed_length, vec_size * dimension_data.length,
                            vec_size * dimension_data.committed_length, 0,
                            vec_size * i * dimension_data.committed_length + static_cast<std::size_t>(output_offset),
                            num_copies, storage, desc.queue)
            .wait();
      } else {
        detail::global_impl_driver<SubgroupSize>(
            in, in_imag, out, out_imag, desc, dimension_data, kernels, dimension_data.transpose_kernels,
            dimension_data.num_forward_factors, 0, dimension_data.forward_impl_twiddle_offset, 0, i,
            static_cast<IdxGlobal>(num_batches),
            static_cast<IdxGlobal>(vec_size * i * dimension_data.length) + input_offset,
            static_cast<IdxGlobal>(vec_size * i * dimension_data.length) + output_offset, storage,
            detail::elementwise_multiply::NOT_APPLIED, static_cast<const Scalar*>(nullptr));
      }
    }
    return event;
  }
};
}  // namespace detail
}  // namespace portfft

#endif
