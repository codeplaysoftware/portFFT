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

#ifndef PORTFFT_COMMON_TRANSPOSE_HPP
#define PORTFFT_COMMON_TRANSPOSE_HPP

#include <sycl/sycl.hpp>

#include "portfft/defines.hpp"

namespace portfft {
namespace detail {
/**
 * Implements Tiled transpose for complex inputs of arbitrary size in global memory.
 * Assumes the input in INTERLEAVED_COMPLEX storage. Works out of place
 *
 * @tparam VecSize Size of each matrix element
 * @tparam T Scalar input type
 * @param N Number of input rows
 * @param M Number of input columns
 * @param tile_size Tile Size
 * @param input Input pointer
 * @param output Output Pointer
 * @param loc 2D local memory accessor of size {tile_size, VecSize * tile_size}
 * @param global_data global data for the kernel
 */
template <int VecSize = 2, typename T>
PORTFFT_INLINE inline void generic_transpose(IdxGlobal N, IdxGlobal M, Idx tile_size, const T* input, T* output,
                                             const sycl::local_accessor<T, 2>& loc,
                                             detail::global_data_struct<2> global_data) {
  static_assert(VecSize <= 2, "VecSize must be either 1 or 2.");
  using T_vec = sycl::vec<T, VecSize>;
  T_vec priv;
  IdxGlobal rounded_up_n = detail::round_up_to_multiple(N, static_cast<IdxGlobal>(tile_size));
  IdxGlobal rounded_up_m = detail::round_up_to_multiple(M, static_cast<IdxGlobal>(tile_size));
  global_data.log_message_global("VecSize: ", VecSize);
  global_data.log_message_global(__func__, "Entered transpose function with lda: ", M, "ldb: ", N,
                                 "which are rounded up to: ", rounded_up_n, ", ", rounded_up_m);
  IdxGlobal start_y = static_cast<IdxGlobal>(global_data.it.get_group(1));
  IdxGlobal y_increment = static_cast<IdxGlobal>(global_data.it.get_group_range(1));
  IdxGlobal start_x = static_cast<IdxGlobal>(global_data.it.get_group(0));
  IdxGlobal x_increment = static_cast<IdxGlobal>(global_data.it.get_group_range(0));
  IdxGlobal tid_y = static_cast<IdxGlobal>(global_data.it.get_local_id(1));
  IdxGlobal tid_x = static_cast<IdxGlobal>(global_data.it.get_local_id(0));

  for (IdxGlobal tile_y = start_y; tile_y < rounded_up_n; tile_y += y_increment) {
    for (IdxGlobal tile_x = start_x; tile_x < rounded_up_m; tile_x += x_increment) {
      IdxGlobal tile_id_y = tile_y * static_cast<IdxGlobal>(tile_size);
      IdxGlobal tile_id_x = tile_x * static_cast<IdxGlobal>(tile_size);

      IdxGlobal i = tile_id_y + tid_y;
      IdxGlobal j = tile_id_x + tid_x;

      if (i < N && j < M) {
        priv.load(0, detail::get_global_multi_ptr(&input[VecSize * (i * M + j)]));
        loc[global_data.it.get_local_id(0)][VecSize * global_data.it.get_local_id(1)] = priv[0];
        global_data.log_message(__func__, "loaded data", priv, "from global index: ", VecSize * (i * M + j),
                                " and storing it to local index: ", global_data.it.get_local_id(0), ", ",
                                VecSize * global_data.it.get_local_id(1));
        if constexpr (VecSize > 1) {
          loc[global_data.it.get_local_id(0)][VecSize * global_data.it.get_local_id(1) + 1] = priv[1];
        }
      }
      sycl::group_barrier(global_data.it.get_group());

      IdxGlobal i_transposed = tile_id_x + tid_y;
      IdxGlobal j_transposed = tile_id_y + tid_x;

      if (j_transposed < N && i_transposed < M) {
        priv[0] = loc[global_data.it.get_local_id(1)][VecSize * global_data.it.get_local_id(0)];
        if constexpr (VecSize > 1) {
          priv[1] = loc[global_data.it.get_local_id(1)][VecSize * global_data.it.get_local_id(0) + 1];
        }
        priv.store(0, detail::get_global_multi_ptr(&output[VecSize * i_transposed * N + VecSize * j_transposed]));
        global_data.log_message(__func__, "stored data", priv, "from local index: ", global_data.it.get_local_id(1),
                                ", ", VecSize * global_data.it.get_local_id(0), " and storing it to global index: ",
                                VecSize * i_transposed * N + VecSize * j_transposed);
      }
      // TODO: This barrier should not required, use double buffering. Preferably use portBLAS
      sycl::group_barrier(global_data.it.get_group());
    }
  }
}

template <Idx SubgroupSize, typename T>
PORTFFT_INLINE void shuffle_transpose(T* priv, T* output, Idx lda, Idx ldb, sycl::sub_group sg) {
  Idx sg_local_linear_id = static_cast<Idx>(sg.get_local_linear_id());
  Idx id_of_thread_in_fft = sg_local_linear_id % lda;
  Idx matrix_start_lane_id = (sg_local_linear_id - id_of_thread_in_fft) & (SubgroupSize - 1);
  Idx lane_id_relative_to_start = id_of_thread_in_fft & (lda - 1);

  PORTFFT_UNROLL
  for (Idx id_of_element_in_wi = 0; id_of_element_in_wi < ldb; id_of_element_in_wi++) {
    Idx relative_target_lane_id = ((lane_id_relative_to_start + id_of_element_in_wi) & (ldb - 1)) * (lda / ldb) +
                                  (lane_id_relative_to_start / ldb);
    Idx target_lane_id = matrix_start_lane_id + relative_target_lane_id;
    Idx store_address = (sg_local_linear_id + id_of_element_in_wi) & (ldb - 1);
    Idx target_address = ((ldb - id_of_element_in_wi) + (sg_local_linear_id / (lda / ldb))) & (ldb - 1);
    T& real_value = priv[2 * target_address];
    T& complex_value = priv[2 * target_address + 1];
    output[2 * store_address] = sycl::select_from_group(sg, real_value, static_cast<std::size_t>(target_lane_id));
    output[2 * store_address + 1] =
        sycl::select_from_group(sg, complex_value, static_cast<std::size_t>(target_lane_id));
  }
}

}  // namespace detail
}  // namespace portfft
#endif
