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

#ifndef PORTFFT_TEST_COMMON_SUB_TUPLE_HPP
#define PORTFFT_TEST_COMMON_SUB_TUPLE_HPP

#include <tuple>
#include <type_traits>
#include <utility>

/// Helper function, @see get_sub_tuple
template <typename SubTupleT, typename TupleT, std::size_t... Idx>
SubTupleT sub_tuple_impl(TupleT big, std::index_sequence<Idx...>) {
  static_assert((std::is_same_v<std::tuple_element_t<Idx, SubTupleT>, std::tuple_element_t<Idx, TupleT>> && ...));
  return std::make_tuple(std::get<Idx>(big)...);
}

/**
 * Return a sub-tuple from a given tuple.
 * All the types from the sub-tuple must match with the first types from the original tuple.
 * The remaining values are ignored.
 *
 * @tparam SubTupleT Returned tuple of type std::tuple<T0, ..., T_N>
 * @tparam TupleT Original tuple of type std::tuple<T0, ..., T_N, U_0, ..., U_N>
 * @param bigger Original tuple values
 */
template <typename SubTupleT, typename TupleT>
SubTupleT get_sub_tuple(TupleT bigger) {
  return sub_tuple_impl<SubTupleT>(bigger, std::make_index_sequence<std::tuple_size<SubTupleT>::value>{});
}

#endif
