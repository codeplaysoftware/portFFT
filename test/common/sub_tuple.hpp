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

template <int I, typename SubTupleT, typename TupleT>
struct sub_tuple;

/// Recursive struct to extract a sub-tuple.
template <int I, typename T, typename... SubTupleTypes, typename TupleT>
struct sub_tuple<I, std::tuple<T, SubTupleTypes...>, TupleT> {
  static_assert(std::is_same_v<T, std::tuple_element_t<I, TupleT>>, "Mismatching sub-tuple types");
  std::tuple<T, SubTupleTypes...> get(TupleT t) {
    return std::tuple_cat(std::make_tuple(std::get<I>(t)),
                          sub_tuple<I + 1, std::tuple<SubTupleTypes...>, TupleT>().get(t));
  }
};

/// Terminating case.
/// Workaround as std::tuple_cat cannot concatenate empty tuples.
template <int I, typename T, typename TupleT>
struct sub_tuple<I, std::tuple<T>, TupleT> {
  static_assert(std::is_same_v<T, std::tuple_element_t<I, TupleT>>, "Mismatching sub-tuple types");
  std::tuple<T> get(TupleT t) { return {std::get<I>(t)}; }
};

/// Extract empty sub-tuple.
template <int I, typename TupleT>
struct sub_tuple<I, std::tuple<>, TupleT> {
  std::tuple<> get(TupleT) { return {}; }
};

/**
 * Return a sub-tuple from a given tuple.
 * All the types from the sub-tuple must match with the first types from the original tuple.
 * The remaining values are ignored.
 *
 * @tparam SubTupleT Returned tuple of type std::tuple<T0, ..., T_N>
 * @tparam TupleT Original tuple of type std::tuple<T0, ..., T_N, U_0, ..., U_N>
 * @param tuple Original tuple values
 */
template <typename SubTupleT, typename TupleT>
SubTupleT get_sub_tuple(TupleT tuple) {
  return sub_tuple<0, SubTupleT, TupleT>().get(tuple);
}

#endif
