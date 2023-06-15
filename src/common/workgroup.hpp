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
 *  Codeplay's SYCL-FFT
 *
 **************************************************************************/

#ifndef SYCL_FFT_COMMON_WORKGROUP_HPP
#define SYCL_FFT_COMMON_WORKGROUP_HPP

#include <common/helpers.hpp>
#include <common/subgroup.hpp>
#include <enums.hpp>

namespace sycl_fft {

/**
 * @brief Implements Subgroup level transpose. Each subgroup can handle more than one nxm matrices, where n,m are
 * arbitrary dimensions, each being lesser than subgroup size. Works out of place.
 *
 * @tparam num_complex_per_wi number of complex values each workitem holds
 * @tparam num_threads_per_fft second dimension of the matrix, number of threads taking part in the transpose
 * @tparam subgroup_size Subgroup size
 * @tparam T_ptr Pointer type to input and output private memory
 *
 * @param priv Input to be transposed
 * @param output Transposed Out
 */
template <int num_complex_per_wi, int num_threads_per_fft, int subgroup_size, typename T_ptr>
__attribute__((always_inline)) inline void transpose(T_ptr priv, T_ptr output, sycl::sub_group sg) {
  using T = detail::remove_ptr<T_ptr>;
  int id_of_thread_in_fft = sg.get_local_linear_id() % num_threads_per_fft;
  int current_simd_lane = sg.get_local_linear_id() & (subgroup_size - 1);
  int batch_start_simd_lane = (sg.get_local_linear_id() - id_of_thread_in_fft) & (subgroup_size - 1);
  int simd_lane_relative_to_batch = id_of_thread_in_fft & (num_threads_per_fft - 1);

  detail::unrolled_loop<0, num_complex_per_wi, 1>([&](const int id_of_element_in_wi) __attribute__((always_inline)) {
    int relative_target_simd_lane = ((simd_lane_relative_to_batch + id_of_element_in_wi) & (num_complex_per_wi - 1)) *
                                        (num_threads_per_fft / num_complex_per_wi) +
                                    (simd_lane_relative_to_batch / num_complex_per_wi);
    int target_simd_lane = batch_start_simd_lane + relative_target_simd_lane;
    int store_address = (current_simd_lane + id_of_element_in_wi) & (num_complex_per_wi - 1);
    int target_address = ((num_complex_per_wi - id_of_element_in_wi) +
                          (current_simd_lane / (num_threads_per_fft / num_complex_per_wi))) &
                         (num_complex_per_wi - 1);
    T& real_value = priv[2 * target_address];
    T& complex_value = priv[2 * target_address + 1];
    output[2 * store_address] = sycl::select_from_group(sg, real_value, target_simd_lane);
    output[2 * store_address + 1] = sycl::select_from_group(sg, complex_value, target_simd_lane);
  });
}

// clang-format off
/**
 * @brief Entire workgroup transposes data in the local memory in place, viewing the data as a N x M Matrix
 * Works by fragmenting the data in the local memory into multiple tiles, transposing each tile, and then transposing
 * the tile arrangement 
 * A B C          A' D'
 *       ------>  B' E'   where each alphabet is a tile of some size
 * D E F          C' F'
 * Transposes tiles row by row
 * @tparam N Number of rows
 * @tparam M Number of columns
 * @tparam num_subgroups number of subgroup working
 * @tparam subgroup_size workitems in each subgroup
 * @tparam T_ptr Pointer to local memory
 *
 * @param loc Pointer to the local memory
 */
// clang-format on
template <int N, int M, int num_subgroups, int subgroup_size, typename T_ptr>
__attribute__((always_inline)) inline void tiled_transpose(T_ptr loc, sycl::nd_item<1> it) {
  using T = detail::remove_ptr<T_ptr>;
  sycl::sub_group sg = it.get_sub_group();
  constexpr int number_of_rows_per_tile = subgroup_size;
  constexpr int num_elements_per_wi = subgroup_size;
  constexpr int num_tiles_along_row = N / number_of_rows_per_tile;
  constexpr int num_tiles_along_col = M / num_elements_per_wi;
  constexpr int tile_stride = num_subgroups;
  int tile_start = sg.get_group_id();

  T input[2 * num_elements_per_wi];
  T output[2 * num_elements_per_wi];

  detail::unrolled_loop<0, num_tiles_along_row, 1>([&](const int i) __attribute__((always_inline)) {
    int row_start_index = 2 * M * i * number_of_rows_per_tile;
    // TODO: use sg.load/store functions ?
    for (int j = tile_start; j < num_tiles_along_col; j += tile_stride) {
      // load the tile
      int row_start_in_tile = row_start_index + sg.get_local_linear_id() * 2 * M;
      int col_start_in_tile = 2 * j * num_elements_per_wi;
      for (int k = 0; k < 2 * num_elements_per_wi; k++) {
        input[k] = loc[detail::pad_local(row_start_in_tile + col_start_in_tile + k)];
      }
      transpose<num_elements_per_wi, subgroup_size, subgroup_size>(input, output, sg);
      for (int k = 0; k < 2 * num_elements_per_wi; k++) {
        loc[detail::pad_local(row_start_in_tile + col_start_in_tile + k)] = output[k];
      }
    }
  });

  sycl::group_barrier(it.get_group());

  detail::unrolled_loop<0, num_tiles_along_row, 1>([&](const int i) __attribute__((always_inline)) {
    int row_start_index = 2 * M * i * number_of_rows_per_tile;
    for (int j = tile_start; j < num_tiles_along_col; j += tile_stride) {
      int row_start_in_tile = row_start_index + sg.get_local_linear_id() * 2 * M;
      int col_start_in_tile = 2 * j * num_elements_per_wi;
      for (int k = 0; k < 2 * num_elements_per_wi; k++) {
        input[k] = loc[detail::pad_local(row_start_in_tile + col_start_in_tile + k)];
      }
    }
  });

  sycl::group_barrier(it.get_group());

  detail::unrolled_loop<0, num_tiles_along_row, 1>([&](const int i) __attribute__((always_inline)) {
    for (int j = tile_start; j < num_tiles_along_col; j += tile_stride) {
      int tile_row_start_index = 2 * M * (j + sg.get_local_linear_id());
      int tile_col_start_index = 2 * i * num_elements_per_wi;
      for (int k = 0; k < 2 * num_elements_per_wi; k++) {
        loc[detail::pad_local(tile_row_start_index + tile_col_start_index + k)] = input[k];
      }
    }
  });
  sycl::group_barrier(it.get_group());
}

template <direction dir, int fact_wi_M, int fact_sg_M, int fact_wi_N, int fact_sg_N, int fft_size, int N, int M,
          typename T_ptr, typename T, typename T_twiddles_ptr, typename twiddles_type>
__attribute__((always_inline)) inline void wg_dft(T_ptr priv, T_ptr scratch, const sycl::local_accessor<T, 1>& loc,
                                                  T_twiddles_ptr loc_twiddles, twiddles_type* wg_twiddles,
                                                  sycl::nd_item<1> it, T scaling_factor, const sycl::stream& out) {
  sycl::sub_group sg = it.get_sub_group();
  constexpr int sg_size = SYCLFFT_TARGET_SUBGROUP_SIZE;
  constexpr int m_ffts_in_sg = sg_size / fact_sg_M;
  constexpr int n_ffts_in_sg = sg_size / fact_sg_N;
  constexpr int m_reals_per_fft = 2 * M;
  constexpr int n_reals_per_fft = 2 * N;
  constexpr int num_threads_per_fft_in_sg_m = m_ffts_in_sg / SYCLFFT_TARGET_SUBGROUP_SIZE;
  constexpr int num_threads_per_fft_in_sg_n = n_ffts_in_sg / SYCLFFT_TARGET_SUBGROUP_SIZE;
  int sg_id = sg.get_group_id();
  constexpr int num_sgs = SYCLFFT_SGS_IN_WG;

  constexpr int max_working_tid_in_sg_m = m_ffts_in_sg * fact_sg_M;
  constexpr int max_working_tid_in_sg_n = n_ffts_in_sg * fact_sg_N;

  int m_sg_offset = sg_id * m_ffts_in_sg + sg.get_local_linear_id() / fact_sg_M;
  int m_sg_increment = num_sgs * m_ffts_in_sg;
  int max_m_sg_offset =
      detail::roundUpToMultiple<size_t>(N, m_ffts_in_sg) + (sg.get_local_linear_id() >= max_working_tid_in_sg_m);

  int n_sg_offset = sg_id * n_ffts_in_sg + sg.get_local_linear_id() / fact_sg_N;
  int n_sg_increment = num_sgs * n_ffts_in_sg;
  int max_n_sg_offset =
      detail::roundUpToMultiple<size_t>(M, n_ffts_in_sg) + (sg.get_local_linear_id() >= max_working_tid_in_sg_n);

  int id_of_wi_in_fft = sg.get_local_linear_id() % fact_sg_M;

  for (int sub_batch = n_sg_offset; sub_batch <= max_n_sg_offset; sub_batch += n_sg_increment) {
    bool working = sub_batch < M && sg.get_local_linear_id() < max_working_tid_in_sg_n;
    if (working) {
      local2private_transposed<fact_wi_N, M, detail::pad::DO_PAD>(loc, priv, sg.get_local_linear_id() % fact_sg_N,
                                                                  sub_batch);
    }

    sg_dft<dir, fact_wi_N, fact_sg_N>(priv, sg, loc_twiddles.get_pointer() + (2 * M));
    transpose<fact_wi_N, fact_sg_N, SYCLFFT_TARGET_SUBGROUP_SIZE>(priv, scratch, sg);

    detail::unrolled_loop<0, fact_wi_N, 1>([&](const int i) __attribute__((always_inline)) {
      T& curr_real = scratch[2 * i];
      T& curr_imag = scratch[2 * i + 1];

      T twiddle_m_index = sub_batch;
      T twiddle_n_index = (sg.get_local_linear_id() % fact_sg_N) * fact_wi_N + i;
      size_t twiddle_index = 2 * M * twiddle_n_index + 2 * twiddle_m_index;
      T twiddle_real = wg_twiddles[twiddle_index];
      T twiddle_imag = wg_twiddles[twiddle_index + 1];
      if (dir == direction::BACKWARD) twiddle_imag = -twiddle_imag;
      T tmp_real = scratch[2 * i];
      curr_real = tmp_real * twiddle_real - curr_imag * twiddle_imag;
      curr_imag = tmp_real * twiddle_imag + curr_imag * twiddle_real;
    });

    if (working) {
      private2local_transposed<fact_wi_N, M, detail::pad::DO_PAD>(loc, scratch, sg.get_local_linear_id() % fact_sg_N,
                                                                  sub_batch);
    }
  }

  sycl::group_barrier(it.get_group());

  for (int sub_batch = m_sg_offset; sub_batch <= max_m_sg_offset; sub_batch += m_sg_increment) {
    bool working = sub_batch < N && sg.get_local_linear_id() < max_working_tid_in_sg_m;
    if (working)
      local2private<2 * fact_wi_M, detail::pad::DO_PAD>(loc, priv, sg.get_local_linear_id() % fact_sg_M, 2 * fact_wi_M,
                                                        2 * M * sub_batch);

    sg_dft<dir, fact_wi_M, fact_sg_M>(priv, sg, loc_twiddles);
    transpose<fact_wi_M, fact_sg_M, SYCLFFT_TARGET_SUBGROUP_SIZE>(priv, scratch, sg);

    detail::unrolled_loop<0, fact_wi_M, 1>([&](const int i) __attribute__((always_inline)) {
      T& curr_real = scratch[2 * i];
      T& curr_imag = scratch[2 * i + 1];

      curr_real *= scaling_factor;
      curr_imag *= scaling_factor;
    });

    if (working) {
      private2local<2 * fact_wi_M, detail::pad::DO_PAD>(scratch, loc, sg.get_local_linear_id() % fact_sg_M,
                                                        2 * fact_wi_M, 2 * M * sub_batch);
    }
  }
  sycl::group_barrier(it.get_group());

  tiled_transpose<N, M, SYCLFFT_SGS_IN_WG, SYCLFFT_TARGET_SUBGROUP_SIZE>(loc, it);
}
}  // namespace sycl_fft
#endif