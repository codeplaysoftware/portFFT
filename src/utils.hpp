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

#ifndef PORTFFT_UTILS_HPP
#define PORTFFT_UTILS_HPP

#include <descriptor.hpp>
#include <enums.hpp>
#include <vector>

namespace portfft {

namespace detail {
/**
 * Get kernel ids for the implementation used.
 *
 * @tparam kernel which base template for kernel to use
 * @tparam SubgroupSize size of the subgroup
 * @param ids vector of kernel ids
 */
template <
    template <typename, domain, direction, detail::memory, detail::transpose, detail::transpose, bool, bool, int, int>
    class Kernel,
    typename Scalar, domain Domain, int SubgroupSize, int kernel_id = 0>
void get_ids(std::vector<sycl::kernel_id>& ids) {
// if not used, some kernels might be optimized away in AOT compilation and not available here
#define PORTFFT_GET_ID(DIRECTION, MEMORY, TRANSPOSE_IN, TRANSPOSE_OUT, LOAD_CALLBACK, STORE_CALLBACK)        \
  try {                                                                                                      \
    ids.push_back(sycl::get_kernel_id<Kernel<Scalar, Domain, DIRECTION, MEMORY, TRANSPOSE_IN, TRANSPOSE_OUT, \
                                             LOAD_CALLBACK, STORE_CALLBACK, SubgroupSize, kernel_id>>());    \
  } catch (...) {                                                                                            \
  }
  // TODO: A better way to do this instead of a long list of all possibilities
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::USM, detail::transpose::TRANSPOSED, detail::transpose::TRANSPOSED,
                 false, false)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::USM, detail::transpose::TRANSPOSED, detail::transpose::TRANSPOSED,
                 false, false)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED,
                 detail::transpose::TRANSPOSED, false, false)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED,
                 detail::transpose::TRANSPOSED, false, false)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::USM, detail::transpose::NOT_TRANSPOSED,
                 detail::transpose::TRANSPOSED, false, false)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::USM, detail::transpose::NOT_TRANSPOSED,
                 detail::transpose::TRANSPOSED, false, false)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::BUFFER, detail::transpose::NOT_TRANSPOSED,
                 detail::transpose::TRANSPOSED, false, false)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::BUFFER, detail::transpose::NOT_TRANSPOSED,
                 detail::transpose::TRANSPOSED, false, false)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::USM, detail::transpose::TRANSPOSED,
                 detail::transpose::NOT_TRANSPOSED, false, false)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::USM, detail::transpose::TRANSPOSED,
                 detail::transpose::NOT_TRANSPOSED, false, false)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED,
                 detail::transpose::NOT_TRANSPOSED, false, false)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED,
                 detail::transpose::NOT_TRANSPOSED, false, false)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::USM, detail::transpose::TRANSPOSED, detail::transpose::TRANSPOSED,
                 true, false)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::USM, detail::transpose::TRANSPOSED, detail::transpose::TRANSPOSED,
                 true, false)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED,
                 detail::transpose::TRANSPOSED, true, false)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED,
                 detail::transpose::TRANSPOSED, true, false)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::USM, detail::transpose::NOT_TRANSPOSED,
                 detail::transpose::TRANSPOSED, true, false)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::USM, detail::transpose::NOT_TRANSPOSED,
                 detail::transpose::TRANSPOSED, true, false)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::BUFFER, detail::transpose::NOT_TRANSPOSED,
                 detail::transpose::TRANSPOSED, true, false)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::BUFFER, detail::transpose::NOT_TRANSPOSED,
                 detail::transpose::TRANSPOSED, true, false)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::USM, detail::transpose::TRANSPOSED,
                 detail::transpose::NOT_TRANSPOSED, true, false)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::USM, detail::transpose::TRANSPOSED,
                 detail::transpose::NOT_TRANSPOSED, true, false)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED,
                 detail::transpose::NOT_TRANSPOSED, true, false)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED,
                 detail::transpose::NOT_TRANSPOSED, true, false)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::USM, detail::transpose::TRANSPOSED, detail::transpose::TRANSPOSED,
                 false, true)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::USM, detail::transpose::TRANSPOSED, detail::transpose::TRANSPOSED,
                 false, true)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED,
                 detail::transpose::TRANSPOSED, false, true)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED,
                 detail::transpose::TRANSPOSED, false, true)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::USM, detail::transpose::NOT_TRANSPOSED,
                 detail::transpose::TRANSPOSED, false, true)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::USM, detail::transpose::NOT_TRANSPOSED,
                 detail::transpose::TRANSPOSED, false, true)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::BUFFER, detail::transpose::NOT_TRANSPOSED,
                 detail::transpose::TRANSPOSED, false, true)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::BUFFER, detail::transpose::NOT_TRANSPOSED,
                 detail::transpose::TRANSPOSED, false, true)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::USM, detail::transpose::TRANSPOSED,
                 detail::transpose::NOT_TRANSPOSED, false, true)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::USM, detail::transpose::TRANSPOSED,
                 detail::transpose::NOT_TRANSPOSED, false, true)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED,
                 detail::transpose::NOT_TRANSPOSED, false, true)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED,
                 detail::transpose::NOT_TRANSPOSED, false, true)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::USM, detail::transpose::TRANSPOSED, detail::transpose::TRANSPOSED,
                 true, true)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::USM, detail::transpose::TRANSPOSED, detail::transpose::TRANSPOSED,
                 true, true)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED,
                 detail::transpose::TRANSPOSED, true, true)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED,
                 detail::transpose::TRANSPOSED, true, true)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::USM, detail::transpose::NOT_TRANSPOSED,
                 detail::transpose::TRANSPOSED, true, true)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::USM, detail::transpose::NOT_TRANSPOSED,
                 detail::transpose::TRANSPOSED, true, true)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::BUFFER, detail::transpose::NOT_TRANSPOSED,
                 detail::transpose::TRANSPOSED, true, true)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::BUFFER, detail::transpose::NOT_TRANSPOSED,
                 detail::transpose::TRANSPOSED, true, true)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::USM, detail::transpose::TRANSPOSED,
                 detail::transpose::NOT_TRANSPOSED, true, true)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::USM, detail::transpose::TRANSPOSED,
                 detail::transpose::NOT_TRANSPOSED, true, true)
  PORTFFT_GET_ID(direction::FORWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED,
                 detail::transpose::NOT_TRANSPOSED, true, true)
  PORTFFT_GET_ID(direction::BACKWARD, detail::memory::BUFFER, detail::transpose::TRANSPOSED,
                 detail::transpose::NOT_TRANSPOSED, true, true)
#undef PORTFFT_GET_ID
}

// template <int kernel_id, typename F, typename G>
// inline void factorize_input(std::size_t input_size, F fits_in_target_level, G select_impl) {

//   if (fits_in_target_level(input_size)) {
//     select_impl.template operator()<kernel_id>(input_size);
//     return;
//   }
//   std::size_t fact_1 = detail::factorize(input_size);
//   if (fits_in_target_level(fact_1)) {
//     select_impl.template operator()<kernel_id>(fact_1);
//   } else {
//     factorize_input<kernel_id>(fact_1, fits_in_target_level, select_impl);
//   }

//   factorize_input<kernel_id + 1>(input_size / fact_1, fits_in_target_level, select_impl);
// }

template <int kernel_id, typename F, typename G>
struct factorize_input_struct {
    static void execute(std::size_t input_size, F fits_in_target_level, G select_impl) {
      if (fits_in_target_level(input_size)) {
        select_impl.template operator()<kernel_id>(input_size);
        return;
      }
      std::size_t fact_1 = detail::factorize(input_size);
      if (fits_in_target_level(fact_1)) {
        select_impl.template operator()<kernel_id>(fact_1);
      } else {
        execute(fact_1, fits_in_target_level, select_impl);
      }

     factorize_input_struct<kernel_id + 1, F, G>::execute(input_size / fact_1, fits_in_target_level, select_impl);
    }
};

template <typename F, typename G>
struct  factorize_input_struct<64, F, G>
{
    static void execute(std::size_t , F , G ) {
      return;
    }
};


}  // namespace detail
}  // namespace portfft

#endif  // PORTFFT_UTILS_HPP