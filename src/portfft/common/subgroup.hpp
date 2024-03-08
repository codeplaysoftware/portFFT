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

#ifndef PORTFFT_COMMON_SUBGROUP_HPP
#define PORTFFT_COMMON_SUBGROUP_HPP

#include <sycl/sycl.hpp>

#include "helpers.hpp"
#include "portfft/common/logging.hpp"
#include "portfft/common/memory_views.hpp"
#include "portfft/common/transfers.hpp"
#include "portfft/common/transpose.hpp"
#include "portfft/defines.hpp"
#include "portfft/enums.hpp"
#include "twiddle.hpp"
#include "twiddle_calc.hpp"
#include "workitem.hpp"

namespace portfft {
namespace detail {

/*
`sg_dft` calculates a DFT by a subgroup on values that are already loaded into private memory of the workitems in the
subgroup. It needs twiddle factors precalculated by `sg_calc_twiddles`. It handles the first factor by cross subgroup
DFT calling `cross_sg_dispatcher` and the second one by workitem implementation - calling `wi_dft`. It does twiddle
multiplication inbetween, but does not transpose. Transposition is supposed to be done when storing the values back to
the local memory.

The size of the DFT performed by this function is `N * M` - for the arguments `N` and `M`. `N` workitems work jointly on
one DFT, so at most `subgroup_size / N` DFTs can be performed by one subgroup at a time. If `N` does not evenly divide
`subgroup_size`, extra workitems perform dummy computations. However, they must also call `sg_dft`, as it uses group
functions.

On input, each of the `N` workitems hold `M` consecutive complex input values. On output, each of the workitems holds
complex values that are strided with stride `N` and consecutive workitems have consecutive values.

`cross_sg_dft` calculates DFT across workitems, with each workitem contributing one complex value as input and output of
the computation. If the size of the subgroup is large enough compared to FFT size, a subgroup can calculate multiple
DFTs at once (the same holds true for `cross_sg_cooley_tukey_dft` and `cross_sg_naive_dft`). It calls either
`cross_sg_cooley_tukey_dft` (for composite sizes) or `cross_sg_naive_dft` (for prime sizes).

`cross_sg_cooley_tukey_dft` calculates DFT of a composite size across workitems. It calls `cross_sg_dft` for each of the
factors and does transposition and twiddle multiplication inbetween.

`cross_sg_naive_dft` calculates DFT across workitems using naive DFT algorithm.
*/

// forward declaration
template <Idx SubgroupSize, Idx RecursionLevel, typename T>
PORTFFT_INLINE void cross_sg_dft(T& real, T& imag, Idx fft_size, Idx stride, sycl::sub_group& sg);

/**
 * Calculates DFT using naive algorithm by using workitems of one subgroup.
 * Each workitem holds one input and one output complex value.
 *
 * @tparam T type of the scalar to work on
 * @param[in,out] real real component of the input/output complex value for one
 * workitem
 * @param[in,out] imag imaginary component of the input/output complex value for
 * one workitem
 * @param fft_size size of the DFT transform
 * @param stride Stride between workitems working on consecutive values of one
 * DFT
 * @param sg subgroup
 */
template <typename T>
PORTFFT_INLINE void cross_sg_naive_dft(T& real, T& imag, Idx fft_size, Idx stride, sycl::sub_group& sg) {
  if (fft_size == 2 && (stride & (stride - 1)) == 0) {
    Idx local_id = static_cast<Idx>(sg.get_local_linear_id());
    Idx idx_out = (local_id / stride) % 2;

    T multi_re = (idx_out & 1) ? T(-1) : T(1);
    T res_real = real * multi_re;
    T res_imag = imag * multi_re;

    res_real += sycl::permute_group_by_xor(sg, real, static_cast<typename sycl::sub_group::linear_id_type>(stride));
    res_imag += sycl::permute_group_by_xor(sg, imag, static_cast<typename sycl::sub_group::linear_id_type>(stride));

    real = res_real;
    imag = res_imag;
  } else {
    Idx local_id = static_cast<Idx>(sg.get_local_linear_id());
    Idx idx_out = (local_id / stride) % fft_size;
    Idx fft_start = local_id - idx_out * stride;

    T res_real = 0;
    T res_imag = 0;

    // IGC doesn't unroll this loop and generates a warning when called from workgroup impl.
    PORTFFT_UNROLL
    for (Idx idx_in = 0; idx_in < fft_size; idx_in++) {
      T multi_re = twiddle<T>::Re[fft_size][idx_in * idx_out % fft_size];
      T multi_im = twiddle<T>::Im[fft_size][idx_in * idx_out % fft_size];

      Idx source_wi_id = fft_start + idx_in * stride;

      T cur_real = sycl::select_from_group(sg, real, static_cast<std::size_t>(source_wi_id));
      T cur_imag = sycl::select_from_group(sg, imag, static_cast<std::size_t>(source_wi_id));

      // multiply cur and multi
      T tmp_real;
      T tmp_imag;
      detail::multiply_complex(cur_real, cur_imag, multi_re, multi_im, tmp_real, tmp_imag);
      res_real += tmp_real;
      res_imag += tmp_imag;
    }

    real = res_real;
    imag = res_imag;
  }
}

/**
 * Transposes values held by workitems of a subgroup. Transposes rectangles of
 * size N*M. Each of the rectangles can be strided.
 *
 * @tparam T type of the scalar to work on
 * @param[in,out] real real component of the input/output complex value for one
 * workitem
 * @param[in,out] imag imaginary component of the input/output complex value for
 * one workitem
 * @param factor_n inner - contiguous size on input, outer size on output
 * @param factor_m outer size on input, inner - contiguous size on output
 * @param stride Stride between consecutive values of one rectangle
 * @param sg subgroup
 */
template <typename T>
PORTFFT_INLINE void cross_sg_transpose(T& real, T& imag, Idx factor_n, Idx factor_m, Idx stride, sycl::sub_group& sg) {
  Idx local_id = static_cast<Idx>(sg.get_local_linear_id());
  Idx index_in_outer_dft = (local_id / stride) % (factor_n * factor_m);
  Idx k = index_in_outer_dft % factor_n;  // index in the contiguous factor/fft
  Idx n = index_in_outer_dft / factor_n;  // index of the contiguous factor/fft
  Idx fft_start = local_id - index_in_outer_dft * stride;
  Idx source_wi_id = fft_start + stride * (k * factor_m + n);
  real = sycl::select_from_group(sg, real, static_cast<std::size_t>(source_wi_id));
  imag = sycl::select_from_group(sg, imag, static_cast<std::size_t>(source_wi_id));
}

/**
 * Calculates DFT using Cooley-Tukey FFT algorithm. Size of the problem is N*M.
 * Each workitem holds one input and one output complex value.
 *
 * @tparam SubgroupSize Size of subgroup in kernel
 * @tparam RecursionLevel level of recursion in SG dft
 * @tparam T type of the scalar to work on
 * @param[in,out] real real component of the input/output complex value for one
 * workitem
 * @param[in,out] imag imaginary component of the input/output complex value for
 * one workitem
 * @param factor_n the first factor of the problem size
 * @param factor_m the second factor of the problem size
 * @param stride Stride between workitems working on consecutive values of one
 * DFT
 * @param sg subgroup
 */
template <Idx SubgroupSize, Idx RecursionLevel, typename T>
PORTFFT_INLINE void cross_sg_cooley_tukey_dft(T& real, T& imag, Idx factor_n, Idx factor_m, Idx stride,
                                              sycl::sub_group& sg) {
  Idx local_id = static_cast<Idx>(sg.get_local_linear_id());
  Idx index_in_outer_dft = (local_id / stride) % (factor_n * factor_m);
  Idx k = index_in_outer_dft % factor_n;  // index in the contiguous factor/fft
  Idx n = index_in_outer_dft / factor_n;  // index of the contiguous factor/fft

  // factor N
  cross_sg_dft<SubgroupSize, RecursionLevel>(real, imag, factor_n, factor_m * stride, sg);
  // transpose
  cross_sg_transpose(real, imag, factor_n, factor_m, stride, sg);
  T multi_re = twiddle<T>::Re[factor_n * factor_m][k * n];
  T multi_im = twiddle<T>::Im[factor_n * factor_m][k * n];
  detail::multiply_complex(real, imag, multi_re, multi_im, real, imag);
  // factor M
  cross_sg_dft<SubgroupSize, RecursionLevel>(real, imag, factor_m, factor_n * stride, sg);
}

/**
 * Calculates DFT using FFT algorithm. Each workitem holds one input and one
 * output complex value.
 *
 * @tparam SubgroupSize Size of subgroup in kernel
 * @tparam RecursionLevel level of recursion in SG dft
 * @tparam T type of the scalar to work on
 * @param[in,out] real real component of the input/output complex value for one
 * workitem
 * @param[in,out] imag imaginary component of the input/output complex value for
 * one workitem
 * @param fft_size Size of the DFT
 * @param stride Stride between workitems working on consecutive values of one
 * DFT
 * @param sg subgroup
 */
template <Idx SubgroupSize, Idx RecursionLevel, typename T>
PORTFFT_INLINE void cross_sg_dft(T& real, T& imag, Idx fft_size, Idx stride, sycl::sub_group& sg) {
  constexpr Idx MaxRecursionLevel = detail::int_log2(SubgroupSize);
  if constexpr (RecursionLevel < MaxRecursionLevel) {
    const Idx f0 = detail::factorize(fft_size);
    if (f0 >= 2 && fft_size / f0 >= 2) {
      cross_sg_cooley_tukey_dft<SubgroupSize, RecursionLevel + 1>(real, imag, fft_size / f0, f0, stride, sg);
    } else {
      cross_sg_naive_dft(real, imag, fft_size, stride, sg);
    }
  }
}

/**
 * Factorizes a number into two factors, so that one of them will maximal below
 or equal to subgroup size.
 * @tparam T type of the number to factorize
 * @param N the number to factorize
 * @param sg_size subgroup size
 * @return the factor below or equal to subgroup size
 */
template <typename T>
PORTFFT_INLINE constexpr T factorize_sg(T N, Idx sg_size) {
  if constexpr (PORTFFT_SLOW_SG_SHUFFLES) {
    return 1;
  } else {
    for (T i = static_cast<T>(sg_size); i > 1; i--) {
      if (N % i == 0) {
        return i;
      }
    }
    return 1;
  }
}

/**
 * Checks whether a problem can be solved with sub-group implementation
 * without reg spilling.
 * @tparam Scalar type of the real scalar used for the computation
 * @param N Size of the problem, in complex values
 * @param sg_size Size of the sub-group
 * @return true if the problem fits in the registers
 */
template <typename Scalar>
constexpr bool fits_in_sg(IdxGlobal N, Idx sg_size) {
  IdxGlobal factor_sg = factorize_sg(N, sg_size);
  IdxGlobal factor_wi = N / factor_sg;
  return fits_in_wi<Scalar>(factor_wi);
}

};  // namespace detail

/**
 * Calculates FFT of size N*M using workitems in a subgroup. Works in place. The
 * end result needs to be transposed when storing it to the local memory!
 *
 * @tparam SubgroupSize Size of subgroup in kernel
 * @tparam T type of the scalar used for computations
 * @param inout pointer to private memory where the input/output data is
 * @param sg subgroup
 * @param factor_wi number of elements per workitem
 * @param factor_sg number of workitems in a subgroup that work on one FFT
 * @param sg_twiddles twiddle factors to use - calculated by sg_calc_twiddles in
 * commit
 * @param private_scratch Scratch memory for wi implementation
 */
template <Idx SubgroupSize, typename T>
PORTFFT_INLINE void sg_dft(T* inout, sycl::sub_group& sg, Idx factor_wi, Idx factor_sg, const T* sg_twiddles,
                           T* private_scratch) {
  Idx idx_of_wi_in_fft = static_cast<Idx>(sg.get_local_linear_id()) % factor_sg;
  // IGC doesn't unroll this loop and generates a warning when called from workgroup impl.
  PORTFFT_UNROLL
  for (Idx idx_of_element_in_wi = 0; idx_of_element_in_wi < factor_wi; idx_of_element_in_wi++) {
    T& real = inout[2 * idx_of_element_in_wi];
    T& imag = inout[2 * idx_of_element_in_wi + 1];

    if (factor_sg > 1) {
      detail::cross_sg_dft<SubgroupSize, 0>(real, imag, factor_sg, 1, sg);
      if (idx_of_element_in_wi > 0) {
        T twiddle_real = sg_twiddles[idx_of_element_in_wi * factor_sg + idx_of_wi_in_fft];
        T twiddle_imag = sg_twiddles[(idx_of_element_in_wi + factor_wi) * factor_sg + idx_of_wi_in_fft];
        detail::multiply_complex(real, imag, twiddle_real, twiddle_imag, real, imag);
      }
    }
  };
  wi_dft<0>(inout, inout, factor_wi, 1, 1, private_scratch);
}

/**
 * Calculates a twiddle factor for subgroup implementation.
 *
 * @tparam T type of the scalar used for computations
 * @param factor_sg number of workitems in a subgroup that work on one FFT
 * @param factor_wi number of elements per workitem
 * @param n index of the twiddle to calculate in the direction of factor_sg
 * @param k index of the twiddle to calculate in the direction of factor_wi
 * @param sg_twiddles destination into which to store the twiddles
 */
template <typename T>
void sg_calc_twiddles(Idx factor_sg, Idx factor_wi, Idx n, Idx k, T* sg_twiddles) {
  std::complex<T> twiddle = detail::calculate_twiddle<T>(n * k, factor_sg * factor_wi);
  sg_twiddles[k * factor_sg + n] = twiddle.real();
  sg_twiddles[(k + factor_wi) * factor_sg + n] = twiddle.imag();
}

template <detail::level Group, std::size_t GlobalDim, std::size_t LocalDim, std::size_t CopyDims, typename T,
          typename LocView>
PORTFFT_INLINE void subgroup_impl_local2global_strided_copy(T* global_ptr, LocView& loc_view,
                                                            std::array<IdxGlobal, GlobalDim> strides_global,
                                                            std::array<Idx, LocalDim> strides_local,
                                                            IdxGlobal offset_global, Idx offset_local,
                                                            std::array<Idx, CopyDims> copy_strides,
                                                            detail::global_data_struct<1> global_data,
                                                            detail::transfer_direction direction) {
  detail::md_view global_md_view{global_ptr, strides_global, offset_global};
  detail::md_view local_md_view{loc_view, strides_local, offset_local};
  if (direction == detail::transfer_direction::GLOBAL_TO_LOCAL) {
    copy_group<Group>(global_data, global_md_view, local_md_view, copy_strides);
  } else if (direction == detail::transfer_direction::LOCAL_TO_GLOBAL) {
    copy_group<Group>(global_data, local_md_view, global_md_view, copy_strides);
  }
}

template <detail::level Group, std::size_t GlobalDim, std::size_t LocalDim, std::size_t CopyDims, typename T,
          typename LocView>
PORTFFT_INLINE void subgroup_impl_local2global_strided_copy(
    T* global_ptr, T* global_imag_ptr, LocView& loc_view, std::array<IdxGlobal, GlobalDim> strides_global,
    std::array<Idx, LocalDim> strides_local, IdxGlobal offset_global, Idx local_offset, Idx local_imag_offset,
    std::array<Idx, CopyDims> copy_strides, detail::global_data_struct<1> global_data,
    detail::transfer_direction direction) {
  detail::md_view global_md_real_view{global_ptr, strides_global, offset_global};
  detail::md_view global_md_imag_view{global_imag_ptr, strides_global, offset_global};
  detail::md_view local_md_real_view{loc_view, strides_local, local_offset};
  detail::md_view local_md_imag_view{loc_view, strides_local, local_offset + local_imag_offset};
  if (direction == detail::transfer_direction::GLOBAL_TO_LOCAL) {
    copy_group<Group>(global_data, global_md_real_view, local_md_real_view, copy_strides);
    copy_group<Group>(global_data, global_md_imag_view, local_md_imag_view, copy_strides);
  } else if (direction == detail::transfer_direction::LOCAL_TO_GLOBAL) {
    copy_group<Group>(global_data, local_md_real_view, global_md_real_view, copy_strides);
    copy_group<Group>(global_data, local_md_imag_view, global_md_imag_view, copy_strides);
  }
}

template <Idx PtrViewNDim, Idx PrivViewNDim, typename IdxType, typename PtrView, typename T>
PORTFFT_INLINE void subgroup_impl_local_private_copy(
    PtrView& ptr_view, PtrView& ptr_imag_view, T* priv,
    std::array<std::array<IdxType, PtrViewNDim>, 2> ptr_view_strides_offsets,
    std::array<std::array<Idx, PrivViewNDim>, 2> priv_view_strides_offsets,
    std::array<std::array<IdxType, PtrViewNDim>, 2> ptr_imag_view_strides_offsets,
    std::array<std::array<Idx, PrivViewNDim>, 2> priv_imag_view_strides_offsets, Idx num_elements_to_copy,
    detail::global_data_struct<1> global_data, detail::transfer_direction direction) {
  detail::strided_view ptr_strided_real_view{ptr_view, std::get<0>(ptr_view_strides_offsets),
                                             std::get<1>(ptr_view_strides_offsets)};
  detail::strided_view ptr_strided_imag_view{ptr_imag_view, std::get<0>(ptr_imag_view_strides_offsets),
                                             std::get<1>(ptr_imag_view_strides_offsets)};
  detail::strided_view priv_strided_real_view{priv, std::get<0>(priv_view_strides_offsets),
                                              std::get<1>(priv_view_strides_offsets)};
  detail::strided_view priv_strided_imag_view{priv, std::get<0>(priv_imag_view_strides_offsets),
                                              std::get<1>(priv_imag_view_strides_offsets)};
  if (direction == detail::transfer_direction::LOCAL_TO_PRIVATE) {
    copy_wi(global_data, ptr_strided_real_view, priv_strided_real_view, num_elements_to_copy);
    copy_wi(global_data, ptr_strided_imag_view, priv_strided_imag_view, num_elements_to_copy);
  } else if (direction == detail::transfer_direction::PRIVATE_TO_LOCAL ||
             direction == detail::transfer_direction::PRIVATE_TO_GLOBAL) {
    copy_wi(global_data, priv_strided_real_view, ptr_strided_real_view, num_elements_to_copy);
    copy_wi(global_data, priv_strided_imag_view, ptr_strided_imag_view, num_elements_to_copy);
  }
}

template <Idx PtrViewNDim, typename IdxType, typename PtrView, typename T>
PORTFFT_INLINE void subgroup_impl_local_private_copy(
    PtrView& ptr_view, T* priv, std::array<std::array<IdxType, PtrViewNDim>, 2> ptr_view_strides_offsets,
    Idx num_elements_to_copy, detail::global_data_struct<1> global_data, detail::transfer_direction direction) {
  detail::strided_view ptr_strided_view{ptr_view, std::get<0>(ptr_view_strides_offsets),
                                        std::get<1>(ptr_view_strides_offsets)};
  if (direction == detail::transfer_direction::LOCAL_TO_PRIVATE) {
    copy_wi<2>(global_data, ptr_strided_view, priv, num_elements_to_copy);
  } else if (direction == detail::transfer_direction::PRIVATE_TO_LOCAL ||
             direction == detail::transfer_direction::PRIVATE_TO_GLOBAL) {
    copy_wi<2>(global_data, priv, ptr_strided_view, num_elements_to_copy);
  }
}

template <Idx SubgroupSize, typename TIn, typename LocView>
PORTFFT_INLINE void subgroup_impl_bluestein_localglobal_packed_copy(
    TIn* global_ptr, TIn* global_imag_ptr, LocView& loc_view, Idx committed_size, Idx fft_size,
    IdxGlobal global_ptr_offset, Idx loc_offset, Idx local_imag_offset, Idx n_ffts_in_sg, sycl::sub_group& sg,
    complex_storage storage, detail::transfer_direction direction, detail::global_data_struct<1>& global_data) {
  if (storage == complex_storage::INTERLEAVED_COMPLEX) {
    PORTFFT_UNROLL
    for (Idx i = 0; i < n_ffts_in_sg; i++) {
      if (direction == detail::transfer_direction::GLOBAL_TO_LOCAL) {
        global2local<detail::level::SUBGROUP, SubgroupSize>(
            global_data, global_ptr, loc_view, 2 * committed_size,
            static_cast<IdxGlobal>(2 * i * committed_size) + global_ptr_offset, 2 * i * fft_size + loc_offset);
      } else if (direction == detail::transfer_direction::LOCAL_TO_GLOBAL) {
        local2global<detail::level::SUBGROUP, SubgroupSize>(global_data, loc_view, global_ptr, 2 * committed_size,
                                                            2 * i * fft_size + loc_offset,
                                                            global_ptr_offset + 2 * i * committed_size);
      }
    }
  } else {
    PORTFFT_UNROLL
    for (Idx i = 0; i < n_ffts_in_sg; i++) {
      if (direction == detail::transfer_direction::GLOBAL_TO_LOCAL) {
        global2local<detail::level::SUBGROUP, SubgroupSize>(
            global_data, global_ptr, loc_view, committed_size,
            static_cast<IdxGlobal>(i * committed_size) + global_ptr_offset, i * fft_size + loc_offset);
        global2local<detail::level::SUBGROUP, SubgroupSize>(
            global_data, global_imag_ptr, loc_view, committed_size,
            static_cast<IdxGlobal>(i * committed_size) + global_ptr_offset,
            i * fft_size + loc_offset + local_imag_offset);
      } else if (direction == detail::transfer_direction::LOCAL_TO_GLOBAL) {
        local2global<detail::level::SUBGROUP, SubgroupSize>(global_data, loc_view, global_ptr, committed_size,
                                                            i * fft_size + loc_offset,
                                                            global_ptr_offset + i * committed_size);
        local2global<detail::level::SUBGROUP, SubgroupSize>(global_data, loc_view, global_imag_ptr, committed_size,
                                                            i * fft_size + loc_offset + local_imag_offset,
                                                            global_ptr_offset + i * committed_size);
      }
    }
  }

  sycl::group_barrier(sg);
}

template <Idx SubgroupSize, typename T, typename LocView>
PORTFFT_INLINE void sg_dft_compute(T* priv, T* private_scratch, detail::elementwise_multiply apply_load_modifier,
                                   detail::elementwise_multiply apply_store_modifier,
                                   detail::complex_conjugate conjugate_on_load,
                                   detail::complex_conjugate conjugate_on_store,
                                   detail::apply_scale_factor scale_factor_applied, const T* load_modifier_data,
                                   const T* store_modifier_data, LocView& twiddles_loc_view, T scale_factor,
                                   IdxGlobal modifier_start_offset, Idx id_of_wi_in_fft, Idx factor_sg, Idx factor_wi,
                                   sycl::sub_group& sg) {
  using vec2_t = sycl::vec<T, 2>;
  vec2_t modifier_vec;
  if (conjugate_on_load == detail::complex_conjugate::APPLIED) {
    detail::conjugate_inplace(priv, factor_wi);
  }
  if (apply_load_modifier == detail::elementwise_multiply::APPLIED) {
    PORTFFT_UNROLL
    for (Idx j = 0; j < factor_wi; j++) {
      modifier_vec = *reinterpret_cast<const vec2_t*>(
          &load_modifier_data[modifier_start_offset + 2 * factor_wi * id_of_wi_in_fft + 2 * j]);
      detail::multiply_complex(priv[2 * j], priv[2 * j + 1], modifier_vec[0], modifier_vec[1], priv[2 * j],
                               priv[2 * j + 1]);
    }
  }
  sg_dft<SubgroupSize>(priv, sg, factor_wi, factor_sg, twiddles_loc_view, private_scratch);

  if (conjugate_on_store == detail::complex_conjugate::APPLIED) {
    detail::conjugate_inplace(priv, factor_wi);
  }

  if (apply_store_modifier == detail::elementwise_multiply::APPLIED) {
    PORTFFT_UNROLL
    for (Idx j = 0; j < factor_wi; j++) {
      modifier_vec = *reinterpret_cast<const vec2_t*>(
          &store_modifier_data[modifier_start_offset + 2 * j * factor_sg + 2 * id_of_wi_in_fft]);
      detail::multiply_complex(priv[2 * j], priv[2 * j + 1], modifier_vec[0], modifier_vec[1], priv[2 * j],
                               priv[2 * j + 1]);
    }
  }

  if (scale_factor_applied == detail::apply_scale_factor::APPLIED) {
    PORTFFT_UNROLL
    for (Idx j = 0; j < factor_wi; j++) {
      priv[2 * j] *= scale_factor;
      priv[2 * j + 1] *= scale_factor;
    }
  }
}

template <Idx SubgroupSize, typename T, typename LocTwiddlesView, typename LocView>
PORTFFT_INLINE void sg_bluestein_batch_interleaved(T* priv, T* priv_scratch, LocView& loc_view, const T* load_modifier,
                                                   const T* store_modifier, LocTwiddlesView& twiddles_loc,
                                                   detail::complex_conjugate conjugate_on_load,
                                                   detail::complex_conjugate conjugate_on_store,
                                                   detail::apply_scale_factor scale_applied, T scale_factor,
                                                   Idx id_of_wi_in_fft, Idx factor_sg, Idx factor_wi,
                                                   complex_storage storage, bool wi_working, Idx local_imag_offset,
                                                   Idx max_num_batches_local_mem, Idx fft_idx_in_local,
                                                   sycl::sub_group& sg, detail::global_data_struct<1>& global_data) {
  sg_dft_compute<SubgroupSize>(
      priv, priv_scratch, detail::elementwise_multiply::APPLIED, detail::elementwise_multiply::APPLIED,
      conjugate_on_load, detail::complex_conjugate::NOT_APPLIED, detail::apply_scale_factor::NOT_APPLIED, load_modifier,
      store_modifier, twiddles_loc, scale_factor, 0, id_of_wi_in_fft, factor_sg, factor_wi, sg);

  PORTFFT_UNROLL
  for (Idx i = 0; i < 2 * factor_wi; i++) {
    priv[i] = (priv[i] / (static_cast<T>(factor_sg * factor_wi)));
  }

  if (wi_working) {
    // Store back to local memory only
    if (storage == complex_storage::INTERLEAVED_COMPLEX) {
      subgroup_impl_local_private_copy<2, Idx>(
          loc_view, priv, {{{factor_sg, max_num_batches_local_mem}, {2 * id_of_wi_in_fft, 2 * fft_idx_in_local}}},
          factor_wi, global_data, detail::transfer_direction::PRIVATE_TO_LOCAL);
    } else {
      subgroup_impl_local_private_copy<2, 1, Idx>(
          loc_view, loc_view, priv, {{{factor_sg, max_num_batches_local_mem}, {id_of_wi_in_fft, fft_idx_in_local}}},
          {{{2}, {0}}},
          {{{factor_sg, max_num_batches_local_mem}, {id_of_wi_in_fft, fft_idx_in_local + local_imag_offset}}},
          {{{2}, {1}}}, factor_wi, global_data, detail::transfer_direction::PRIVATE_TO_LOCAL);
    }
  }

  sycl::group_barrier(sg);
  if (wi_working) {
    if (storage == complex_storage::INTERLEAVED_COMPLEX) {
      const Idx fft_element = 2 * id_of_wi_in_fft * factor_wi;
      subgroup_impl_local_private_copy<1, Idx>(
          loc_view, priv,
          {{{max_num_batches_local_mem}, {fft_element * max_num_batches_local_mem + 2 * fft_idx_in_local}}}, factor_wi,
          global_data, detail::transfer_direction::LOCAL_TO_PRIVATE);
    } else {
      subgroup_impl_local_private_copy<2, 1, Idx>(
          loc_view, loc_view, priv, {{{1, max_num_batches_local_mem}, {id_of_wi_in_fft * factor_wi, fft_idx_in_local}}},
          {{{2}, {0}}},
          {{{1, max_num_batches_local_mem}, {id_of_wi_in_fft * factor_wi, fft_idx_in_local + local_imag_offset}}},
          {{{2}, {1}}}, factor_wi, global_data, detail::transfer_direction::LOCAL_TO_PRIVATE);
    }
  }

  auto conjugate_on_output = conjugate_on_store == detail::complex_conjugate::APPLIED
                                 ? detail::complex_conjugate::NOT_APPLIED
                                 : detail::complex_conjugate::APPLIED;

  sg_dft_compute<SubgroupSize>(priv, priv_scratch, detail::elementwise_multiply::NOT_APPLIED,
                               detail::elementwise_multiply::APPLIED, detail::complex_conjugate::APPLIED,
                               conjugate_on_output, scale_applied, static_cast<const T*>(nullptr), load_modifier,
                               twiddles_loc, scale_factor, 0, id_of_wi_in_fft, factor_sg, factor_wi, sg);
}

template <Idx SubgroupSize, typename T, typename LocTwiddlesView, typename LocView>
void sg_bluestein(T* priv, T* priv_scratch, LocView& loc_view, LocTwiddlesView& loc_twiddles, const T* load_modifier,
                  const T* store_modifier, detail::complex_conjugate conjugate_on_load,
                  detail::complex_conjugate conjugate_on_store, detail::apply_scale_factor scale_applied,
                  T scale_factor, Idx id_of_wi_in_fft, Idx factor_sg, Idx factor_wi, complex_storage storage,
                  bool wi_working, Idx loc_offset_store_view, Idx loc_offset_load_view, Idx local_imag_offset,
                  sycl::sub_group sg, detail::global_data_struct<1>& global_data) {
  // for (Idx i = 0; i < 2 * factor_wi; i++) {
  //   priv[i] = 2;
  // }
  sg_dft_compute<SubgroupSize>(
      priv, priv_scratch, detail::elementwise_multiply::APPLIED, detail::elementwise_multiply::APPLIED,
      conjugate_on_load, detail::complex_conjugate::NOT_APPLIED, detail::apply_scale_factor::NOT_APPLIED, load_modifier,
      store_modifier, loc_twiddles, scale_factor, 0, id_of_wi_in_fft, factor_sg, factor_wi, sg);

  PORTFFT_UNROLL
  for (Idx i = 0; i < 2 * factor_wi; i++) {
    priv[i] = (priv[i] / (static_cast<T>(factor_sg * factor_wi)));
  }

  if (wi_working) {
    if (storage == complex_storage::INTERLEAVED_COMPLEX) {
      subgroup_impl_local_private_copy<1, Idx>(loc_view, priv, {{{factor_sg}, {loc_offset_store_view}}}, factor_wi,
                                               global_data, detail::transfer_direction::PRIVATE_TO_LOCAL);
    } else {
      detail::strided_view priv_real_view{priv, 2};
      detail::strided_view priv_imag_view{priv, 2, 1};
      detail::strided_view local_real_view{loc_view, factor_sg, loc_offset_store_view};
      detail::strided_view local_imag_view{loc_view, factor_sg, loc_offset_store_view + local_imag_offset};
      copy_wi(global_data, priv_real_view, local_real_view, factor_wi);
      copy_wi(global_data, priv_imag_view, local_imag_view, factor_wi);
    }
  }

  sycl::group_barrier(sg);

  if (wi_working) {
    if (storage == complex_storage::INTERLEAVED_COMPLEX) {
      subgroup_impl_local_private_copy<1, Idx>(loc_view, priv, {{{1}, {loc_offset_load_view}}}, factor_wi, global_data,
                                               detail::transfer_direction::LOCAL_TO_PRIVATE);
    } else {
      subgroup_impl_local_private_copy<1, 1, Idx>(loc_view, loc_view, priv, {{{1}, {loc_offset_load_view}}},
                                                  {{{2}, {0}}}, {{{1}, {loc_offset_load_view + local_imag_offset}}},
                                                  {{{2}, {1}}}, factor_wi, global_data,
                                                  detail::transfer_direction::LOCAL_TO_PRIVATE);
    }
  }

  auto conjugate_on_output = conjugate_on_store == detail::complex_conjugate::APPLIED
                                 ? detail::complex_conjugate::NOT_APPLIED
                                 : detail::complex_conjugate::APPLIED;

  sg_dft_compute<SubgroupSize>(priv, priv_scratch, detail::elementwise_multiply::NOT_APPLIED,
                               detail::elementwise_multiply::APPLIED, detail::complex_conjugate::APPLIED,
                               conjugate_on_output, scale_applied, static_cast<const T*>(nullptr), load_modifier,
                               loc_twiddles, scale_factor, 0, id_of_wi_in_fft, factor_sg, factor_wi, sg);
}

};  // namespace portfft

#endif
