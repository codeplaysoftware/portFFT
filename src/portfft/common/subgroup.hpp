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
#include "portfft/common/transfers.hpp"
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

/**
 * Function to copy data between local and global memory as required by the subgroup level Bluestein algorithm,
 * when the data in both local and global memory is in packed format,when the storage scheme is INTERLEAVED_COMPLEX
 *
 * @tparam SubgroupSize Subgroup size
 * @tparam Direction  Direction Direction of the copy, expected to be either transfer_direction::LOCAL_TO_GLOBAL or
 * transfer_direction::GLOBAL_TO_LOCAL
 * @tparam TIn Global memory Type
 * @tparam LocView Type of the view constructed for local memory
 * @param global_ptr global memory pointer
 * @param loc_view View of the local memory
 * @param committed_size Size of the DFT as committed, also the number of complex elements in each transform present in
 * global memory
 * @param fft_size The padded DFT size, also the number of elements of complex elements in each transform that resides
 * in local memory
 * @param global_ptr_offset Offset to be applied to the global memory pointer
 * @param loc_offset Offset to be applied to the local memory view
 * @param n_ffts_in_sg Number of ffts that can be calculated by a single subgroup
 * @param transform_id Id of the transform in the kernel
 * @param n_transforms Total number of transforms in the kernel
 * @param global_data global_data_struct associated with the kernel launch
 */
template <Idx SubgroupSize, detail::transfer_direction Direction, typename TIn, typename LocView>
PORTFFT_INLINE void subgroup_impl_bluestein_local_global_packed_copy(
    TIn global_ptr, LocView& loc_view, Idx committed_size, Idx fft_size, IdxGlobal global_ptr_offset, Idx loc_offset,
    Idx n_ffts_in_sg, IdxGlobal transform_id, IdxGlobal n_transforms, detail::global_data_struct<1>& global_data) {
  PORTFFT_UNROLL
  for (Idx i = 0; i < n_ffts_in_sg && ((i + transform_id) < n_transforms); i++) {
    local_global_packed_copy<detail::level::SUBGROUP, Direction, SubgroupSize>(
        global_ptr, loc_view, global_ptr_offset + static_cast<IdxGlobal>(2 * i * committed_size),
        2 * i * fft_size + loc_offset, 2 * committed_size, global_data);
  }
}

/**
 * Function to copy data between local and global memory as required by the subgroup level Bluestein algorithm,
 * when the data in both local and global memory is in packed format,when the storage scheme is SPLIT_COMPLEX
 *
 * @tparam SubgroupSize Subgroup size
 * @tparam Direction  Direction Direction of the copy, expected to be either transfer_direction::LOCAL_TO_GLOBAL or
 * transfer_direction::GLOBAL_TO_LOCAL
 * @tparam TIn Global memory Type
 * @tparam LocView Type of the view constructed for local memory
 * @param global_ptr global memory pointer containing the real part of the data
 * @param global_imag_ptr global memory pointer containing the imaginary  part of the data
 * @param loc_view View of the local memory
 * @param committed_size Size of the DFT as committed, also the number of complex elements in each transform present in
 * global memory
 * @param fft_size The padded DFT size, also the number of elements of complex elements in each transform that resides
 * in local memory
 * @param global_ptr_offset Offset to be applied to the global memory pointer
 * @param loc_offset Offset to be applied to the local memory view
 * @param local_imag_offset Number of elements in local memory after which the imaginary component of the values is
 * stored
 * @param n_ffts_in_sg Number of ffts that can be calculated by a single subgroup
 * @param transform_id Id of the transform in the kernel
 * @param n_transforms Total number of transforms in the kernel
 * @param global_data global_data_struct associated with the kernel launch
 */
template <Idx SubgroupSize, detail::transfer_direction Direction, typename TIn, typename LocView>
PORTFFT_INLINE void subgroup_impl_bluestein_local_global_packed_copy(
    TIn global_ptr, TIn global_imag_ptr, LocView& loc_view, Idx committed_size, Idx fft_size,
    IdxGlobal global_ptr_offset, Idx loc_offset, Idx local_imag_offset, Idx n_ffts_in_sg, IdxGlobal transform_id,
    IdxGlobal n_transforms, detail::global_data_struct<1>& global_data) {
  PORTFFT_UNROLL
  for (Idx i = 0; i < n_ffts_in_sg && (i + transform_id < n_transforms); i++) {
    local_global_packed_copy<detail::level::SUBGROUP, Direction, SubgroupSize>(
        global_ptr, global_imag_ptr, loc_view, static_cast<IdxGlobal>(i * committed_size) + global_ptr_offset,
        i * fft_size + loc_offset, local_imag_offset, committed_size, global_data);
  }
}

/**
 * Performs all the computations to be done in the private memory for the subgroup level FFT Implementation
 *
 * @tparam SubgroupSize Subgroup Size
 * @tparam T Scalar Type
 * @tparam LocView View of the local memory
 * @param priv private memory array on which the computations will be done
 * @param private_scratch Scratch private memory to be passed to the wi_dft as a part of sg_dft
 * @param apply_load_modifier Whether or not modifiers need to be applied before the fft computation
 * @param apply_store_modifier Whether or not the modifiers need to be applied after the fft computation
 * @param conjugate_on_load Whether or not conjugation of the input is to be done before the fft computation
 * @param conjugate_on_store Whether or not conjugation of the input is to be done after the fft computation
 * @param scale_factor_applied Whether or not scale factor is applied
 * @param load_modifier_data Global memory pointer containing the load modifier data, assumed aligned to at least
 * sycl::vec<T, 2>
 * @param store_modifier_data Global memory pointer containing the store modifier data, assumed aligned to at least
 * sycl::vec<T, 2>
 * @param twiddles_loc_view View of the local memory containing the twiddles
 * @param scale_factor Value of the scale factor
 * @param modifier_start_offset offset to be applied to the load/store modifier pointers
 * @param id_of_wi_in_fft workitem id withing the fft
 * @param factor_sg Number of workitems participating for one transform
 * @param factor_wi Number of complex elements per workitem for each transform
 * @param global_data global_data_struct associated with the kernel launch
 */
template <Idx SubgroupSize, typename T, typename LocView>
PORTFFT_INLINE void sg_dft_compute(T* priv, T* private_scratch, detail::elementwise_multiply apply_load_modifier,
                                   detail::elementwise_multiply apply_store_modifier,
                                   detail::complex_conjugate conjugate_on_load,
                                   detail::complex_conjugate conjugate_on_store,
                                   detail::apply_scale_factor scale_factor_applied, const T* load_modifier_data,
                                   const T* store_modifier_data, LocView& twiddles_loc_view, T scale_factor,
                                   IdxGlobal modifier_start_offset, Idx id_of_wi_in_fft, Idx factor_sg, Idx factor_wi,
                                   detail::global_data_struct<1>& global_data) {
  using vec2_t = sycl::vec<T, 2>;
  vec2_t modifier_vec;
  if (conjugate_on_load == detail::complex_conjugate::APPLIED) {
    global_data.log_message(__func__, "Applying complex conjugate before computation of the FFT");
    detail::conjugate_inplace(priv, factor_wi);
  }
  if (apply_load_modifier == detail::elementwise_multiply::APPLIED) {
    global_data.log_message(__func__, "Applying load modifiers");
    PORTFFT_UNROLL
    for (Idx j = 0; j < factor_wi; j++) {
      modifier_vec = *reinterpret_cast<const vec2_t*>(
          &load_modifier_data[modifier_start_offset + 2 * factor_wi * id_of_wi_in_fft + 2 * j]);
      detail::multiply_complex(priv[2 * j], priv[2 * j + 1], modifier_vec[0], modifier_vec[1], priv[2 * j],
                               priv[2 * j + 1]);
    }
  }
  sg_dft<SubgroupSize>(priv, global_data.sg, factor_wi, factor_sg, twiddles_loc_view, private_scratch);

  if (conjugate_on_store == detail::complex_conjugate::APPLIED) {
    global_data.log_message(__func__, "Applying complex conjugate after computation of the FFT");
    detail::conjugate_inplace(priv, factor_wi);
  }

  if (apply_store_modifier == detail::elementwise_multiply::APPLIED) {
    global_data.log_message(__func__, "Applying store modifiers");
    PORTFFT_UNROLL
    for (Idx j = 0; j < factor_wi; j++) {
      modifier_vec = *reinterpret_cast<const vec2_t*>(
          &store_modifier_data[modifier_start_offset + 2 * j * factor_sg + 2 * id_of_wi_in_fft]);
      detail::multiply_complex(priv[2 * j], priv[2 * j + 1], modifier_vec[0], modifier_vec[1], priv[2 * j],
                               priv[2 * j + 1]);
    }
  }

  if (scale_factor_applied == detail::apply_scale_factor::APPLIED) {
    global_data.log_message(__func__, "Applying scale factor");
    PORTFFT_UNROLL
    for (Idx j = 0; j < factor_wi; j++) {
      priv[2 * j] *= scale_factor;
      priv[2 * j + 1] *= scale_factor;
    }
  }
}

/**
 * Implements the Subgroup level Bluestein algorithm with an addition trip to local memory, when the layout of the data
 * in local memory is in BATCH_INTERLEAVED format
 *
 * @tparam SubgroupSize Subgroup Size
 * @tparam T Scalar Type
 * @tparam LocTwiddlesView Type of view of the local memory containing the twiddles
 * @tparam LocView Type of view of the local memory which stores the data
 * @param priv private memory array on which the computations will be done
 * @param private_scratch Scratch private memory to be passed to the wi_dft as a part of sg_dft
 * @param loc_view view of the local memory to store the data
 * @param load_modifier Global memory pointer containing the load modifier data, assumed aligned to at least
 * sycl::vec<T, 2>
 * @param store_modifier Global memory pointer containing the store modifier data, assumed aligned to at least
 * sycl::vec<T, 2>
 * @param twiddles_loc view of the local memory containing the twiddles
 * @param conjugate_on_load Whether or not conjugation of the input is to be done before the fft computation
 * @param conjugate_on_store Whether or not conjugation of the input is to be done after the fft computation
 * @param scale_applied Whether or not scale factor is applied
 * @param scale_factor Value of the scaling factor
 * @param id_of_wi_in_fft Id of the workitem in the FFT
 * @param factor_sg Number of workitems participating for one transform
 * @param factor_wi Number of complex elements per workitem for each transform
 * @param storage storage scheme of complex values in local memory, SPLIT_COMPLEX or INTERLEAVED_COMPLEX
 * @param wi_working Whether or not the workitem participates in the data transfers
 * @param local_imag_offset Number of elements in local memory after which the imaginary component of the values is
 * stored
 * @param max_num_batches_local_mem Maximum number of transforms that can be stored in local memory
 * @param fft_idx_in_local Id of the transform in local memory
 * @param global_data global_data_struct associated with kernel launch
 */
template <Idx SubgroupSize, typename T, typename LocTwiddlesView, typename LocView>
PORTFFT_INLINE void sg_bluestein_batch_interleaved(
    T* priv, T* priv_scratch, LocView& loc_view, const T* load_modifier, const T* store_modifier,
    LocTwiddlesView& twiddles_loc, detail::complex_conjugate conjugate_on_load,
    detail::complex_conjugate conjugate_on_store, detail::apply_scale_factor scale_applied, T scale_factor,
    Idx id_of_wi_in_fft, Idx factor_sg, Idx factor_wi, complex_storage storage, bool wi_working, Idx local_imag_offset,
    Idx max_num_batches_local_mem, Idx fft_idx_in_local, detail::global_data_struct<1>& global_data) {
  global_data.log_message_global(__func__, "computing forward FFT and applying scaling factor for the backward phase");
  sg_dft_compute<SubgroupSize>(
      priv, priv_scratch, detail::elementwise_multiply::APPLIED, detail::elementwise_multiply::APPLIED,
      conjugate_on_load, detail::complex_conjugate::NOT_APPLIED, detail::apply_scale_factor::APPLIED, load_modifier,
      store_modifier, twiddles_loc, static_cast<T>(1. / (static_cast<T>(factor_sg * factor_wi))), 0, id_of_wi_in_fft,
      factor_sg, factor_wi, global_data);

  // TODO: Currently local memory is being used to load the data back in natural order for the backward phase, as result
  // of sg_dft is transposed However, the Ideal way to this is using shuffles. Implement a batched matrix transpose to
  // transpose a matrix stored in the private memory of workitems of a subgroup using shuffles only This we way can even
  // the 2 sg_bluestein functions that we have today
  if (wi_working) {
    global_data.log_message(__func__, "storing result of the forward phase back to local memory");
    if (storage == complex_storage::INTERLEAVED_COMPLEX) {
      local_private_strided_copy<2, Idx>(
          loc_view, priv, {{{factor_sg, max_num_batches_local_mem}, {2 * id_of_wi_in_fft, 2 * fft_idx_in_local}}},
          factor_wi, detail::transfer_direction::PRIVATE_TO_LOCAL, global_data);
    } else {
      local_private_strided_copy<2, Idx>(
          loc_view, loc_view, priv, {{{factor_sg, max_num_batches_local_mem}, {id_of_wi_in_fft, fft_idx_in_local}}},
          {{{factor_sg, max_num_batches_local_mem}, {id_of_wi_in_fft, fft_idx_in_local + local_imag_offset}}},
          factor_wi, detail::transfer_direction::PRIVATE_TO_LOCAL, global_data);
    }
  }

  sycl::group_barrier(global_data.sg);
  if (wi_working) {
    if (storage == complex_storage::INTERLEAVED_COMPLEX) {
      global_data.log_message(__func__, "loading back the result from local memory for the backward phase");
      const Idx fft_element = 2 * id_of_wi_in_fft * factor_wi;
      local_private_strided_copy<1, Idx>(
          loc_view, priv,
          {{{max_num_batches_local_mem}, {fft_element * max_num_batches_local_mem + 2 * fft_idx_in_local}}}, factor_wi,
          detail::transfer_direction::LOCAL_TO_PRIVATE, global_data);
    } else {
      local_private_strided_copy<2, Idx>(
          loc_view, loc_view, priv, {{{1, max_num_batches_local_mem}, {id_of_wi_in_fft * factor_wi, fft_idx_in_local}}},
          {{{1, max_num_batches_local_mem}, {id_of_wi_in_fft * factor_wi, fft_idx_in_local + local_imag_offset}}},
          factor_wi, detail::transfer_direction::LOCAL_TO_PRIVATE, global_data);
    }
  }
  global_data.log_message(__func__, "computing backward FFT and applying user provided scale value");
  sg_dft_compute<SubgroupSize>(priv, priv_scratch, detail::elementwise_multiply::NOT_APPLIED,
                               detail::elementwise_multiply::APPLIED, detail::complex_conjugate::APPLIED,
                               detail::complex_conjugate::APPLIED, scale_applied, static_cast<const T*>(nullptr),
                               load_modifier, twiddles_loc, scale_factor, 0, id_of_wi_in_fft, factor_sg, factor_wi,
                               global_data);

  if (conjugate_on_store == detail::complex_conjugate::APPLIED) {
    global_data.log_message(__func__, "Applying complex conjugate on the output");
    detail::conjugate_inplace(priv, factor_wi);
  }
}

/**
 *
 * Implements the Subgroup level Bluestein algorithm with an addition trip to local memory, when the layout of the data
 * in local memory is in BATCH_INTERLEAVED format
 *
 * @tparam SubgroupSize Subgroup Size
 * @tparam T Scalar Type
 * @tparam LocTwiddlesView Type of view of the local memory containing the twiddles
 * @tparam LocView Type of view of the local memory which stores the data
 * @param priv private memory array on which the computations will be done
 * @param private_scratch Scratch private memory to be passed to the wi_dft as a part of sg_dft
 * @param loc_view view of the local memory to store the data
 * @param load_modifier Global memory pointer containing the load modifier data, assumed aligned to at least
 * sycl::vec<T, 2>
 * @param store_modifier Global memory pointer containing the store modifier data, assumed aligned to at least
 * sycl::vec<T, 2>
 * @param loc_twiddles view of the local memory containing the twiddles
 * @param conjugate_on_load Whether or not conjugation of the input is to be done before the fft computation
 * @param conjugate_on_store Whether or not conjugation of the input is to be done after the fft computation
 * @param scale_applied Whether or not scale factor is applied
 * @param scale_factor Value of the scaling factor
 * @param id_of_wi_in_fft Id of the workitem in the FFT
 * @param factor_sg Number of workitems participating for one transform
 * @param factor_wi Number of complex elements per workitem for each transform
 * @param storage storage scheme of complex values in local memory, SPLIT_COMPLEX or INTERLEAVED_COMPLEX
 * @param wi_working Whether or not the workitem participates in the data transfers
 * @param loc_view_store_offset Offset to be applied to local memory view when storing the data back to local memory
 * after forward fft phase
 * @param loc_view_load_offset offset to be applied to local memory view when loading the data back to local memory for
 * backward fft phase
 * @param local_imag_offset Number of elements in local memory after which the imaginary component of the values is
 * stored
 * @param global_data  global_data_struct associated with kernel launch
 */
template <Idx SubgroupSize, typename T, typename LocTwiddlesView, typename LocView>
void sg_bluestein_packed(T* priv, T* priv_scratch, LocView& loc_view, LocTwiddlesView& loc_twiddles,
                         const T* load_modifier, const T* store_modifier, detail::complex_conjugate conjugate_on_load,
                         detail::complex_conjugate conjugate_on_store, detail::apply_scale_factor scale_applied,
                         T scale_factor, Idx id_of_wi_in_fft, Idx factor_sg, Idx factor_wi, complex_storage storage,
                         bool wi_working, Idx loc_view_store_offset, Idx loc_view_load_offset, Idx local_imag_offset,
                         detail::global_data_struct<1>& global_data) {
  global_data.log_message_global(__func__, "computing forward FFT and applying scaling factor for the backward phase");
  sg_dft_compute<SubgroupSize>(
      priv, priv_scratch, detail::elementwise_multiply::APPLIED, detail::elementwise_multiply::APPLIED,
      conjugate_on_load, detail::complex_conjugate::NOT_APPLIED, detail::apply_scale_factor::APPLIED, load_modifier,
      store_modifier, loc_twiddles, static_cast<T>(1. / static_cast<T>(factor_sg * factor_wi)), 0, id_of_wi_in_fft,
      factor_sg, factor_wi, global_data);

  if (wi_working) {
    global_data.log_message(__func__, "storing result of the forward phase back to local memory");
    if (storage == complex_storage::INTERLEAVED_COMPLEX) {
      local_private_strided_copy<1, Idx>(loc_view, priv, {{{factor_sg}, {loc_view_store_offset}}}, factor_wi,
                                         detail::transfer_direction::PRIVATE_TO_LOCAL, global_data);
    } else {
      local_private_strided_copy<1, Idx>(loc_view, loc_view, priv, {{{factor_sg}, {loc_view_store_offset}}},
                                         {{{factor_sg}, {loc_view_store_offset + local_imag_offset}}}, factor_wi,
                                         detail::transfer_direction::PRIVATE_TO_LOCAL, global_data);
    }
  }

  sycl::group_barrier(global_data.sg);

  if (wi_working) {
    global_data.log_message(__func__, "loading back the result from local memory for the backward phase");
    if (storage == complex_storage::INTERLEAVED_COMPLEX) {
      local_private_strided_copy<1, Idx>(loc_view, priv, {{{1}, {loc_view_load_offset}}}, factor_wi,
                                         detail::transfer_direction::LOCAL_TO_PRIVATE, global_data);
    } else {
      local_private_strided_copy<1, Idx>(loc_view, loc_view, priv, {{{1}, {loc_view_load_offset}}},
                                         {{{1}, {loc_view_load_offset + local_imag_offset}}}, factor_wi,
                                         detail::transfer_direction::LOCAL_TO_PRIVATE, global_data);
    }
  }
  global_data.log_message(__func__, "computing backward FFT and applying user provided scale value");
  sg_dft_compute<SubgroupSize>(priv, priv_scratch, detail::elementwise_multiply::NOT_APPLIED,
                               detail::elementwise_multiply::APPLIED, detail::complex_conjugate::APPLIED,
                               detail::complex_conjugate::APPLIED, scale_applied, static_cast<const T*>(nullptr),
                               load_modifier, loc_twiddles, scale_factor, 0, id_of_wi_in_fft, factor_sg, factor_wi,
                               global_data);
  if (conjugate_on_store == detail::complex_conjugate::APPLIED) {
    global_data.log_message(__func__, "Applying complex conjugate on the output");
    detail::conjugate_inplace(priv, factor_wi);
  }
}

};  // namespace portfft

#endif
