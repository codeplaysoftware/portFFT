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

#ifndef PORTFFT_COMMON_WORKITEM_HPP
#define PORTFFT_COMMON_WORKITEM_HPP

#include <common/helpers.hpp>
#include <common/twiddle.hpp>
#include <enums.hpp>
#include <sycl/sycl.hpp>

namespace portfft {

// forward declaration
template <direction Dir, int WiDftRecursionLevel, typename T>
inline void wi_dft(int dftSize, const T* in, int stride_in, T* out, int stride_out, T* privateScratch);

namespace detail {

// Maximum size of an FFT that can fit in the workitem implementation
static constexpr std::size_t MaxFftSizeWi = 56;

/*
`wi_dft` calculates a DFT by a workitem on values that are already loaded into its private memory.
It calls either `cooley_tukey_dft` (for composite sizes) or `naive_dft` (for prime sizes).

`cooley_tukey_dft` calculates DFT of a composite size by one workitem. It calls `wi_dft` for each of the factors and
does twiddle multiplication in-between. Transposition is handled by calling `wi_dft` with different input and output
strides.

`naive_dft` calculates DFT by one workitem using naive DFT algorithm.
*/

/**
 * Calculates DFT using naive algorithm. Can work in or out of place.
 *
 * @tparam Dir direction of the FFT
 * @tparam T type of the scalar used for computations
 * @param dftSize The size of the DFT
 * @param in pointer to input
 * @param stride_in stride (in complex values) between complex values in `in`
 * @param out pointer to output
 * @param stride_out stride (in complex values) between complex values in `out`
 * @param privateScratch Scratch memory for this WI. Expects 2 * dftSize size.
 */
template <direction Dir, typename T>
__attribute__((always_inline)) inline void naive_dft(int dftSize, const T* in, int stride_in, T* out, int stride_out,
                                                     T* privateScratch) {
#pragma clang loop unroll(full)
  for (int idx_out{0}; idx_out < dftSize; ++idx_out) {
    privateScratch[2 * idx_out + 0] = 0;
    privateScratch[2 * idx_out + 1] = 0;
#pragma clang loop unroll(full)
    for (int idx_in{0}; idx_in < dftSize; ++idx_in) {
      // this multiplier is not really a twiddle factor, but it is calculated the same way
      auto re_multiplier = twiddle<T>::Re[dftSize][idx_in * idx_out % dftSize];
      auto im_multiplier = [&]() {
        if constexpr (Dir == direction::FORWARD) {
          return twiddle<T>::Im[dftSize][idx_in * idx_out % dftSize];
        }
        return -twiddle<T>::Im[dftSize][idx_in * idx_out % dftSize];
      }();

      // multiply in and multi
      privateScratch[2 * idx_out + 0] +=
          in[2 * idx_in * stride_in] * re_multiplier - in[2 * idx_in * stride_in + 1] * im_multiplier;
      privateScratch[2 * idx_out + 1] +=
          in[2 * idx_in * stride_in] * im_multiplier + in[2 * idx_in * stride_in + 1] * re_multiplier;
    }
  }
#pragma clang loop unroll(full)
  for (int idx_out{0}; idx_out < 2 * dftSize; idx_out += 2) {
    out[idx_out * stride_out + 0] = privateScratch[idx_out + 0];
    out[idx_out * stride_out + 1] = privateScratch[idx_out + 1];
  }
}

// mem requirement: ~N*M(if in place, otherwise x2) + N*M(=tmp) + sqrt(N*M) + pow(N*M,0.25) + ...
// TODO explore if this tmp can be reduced/eliminated ^^^^^^
/**
 * Calculates DFT using Cooley-Tukey FFT algorithm. Can work in or out of place. Size of the problem is N*M
 *
 * @tparam Dir direction of the FFT
 * @tparam WiDftRecursionLevel The number of times wi_dft has been called prior to calling this function.
 * @tparam T type of the scalar used for computations
 * @param factorN the first factor of the problem size
 * @param factorM the second factor of the problem size
 * @param in pointer to input
 * @param stride_in stride (in complex values) between complex values in `in`. Expects to be less than factorM.
 * @param out pointer to output
 * @param stride_out stride (in complex values) between complex values in `out`
 * @param privateScratch Scratch memory for this WI. Expects size 2 * (factorN * factorM + max(factorN, factorM)).
 */
template <direction Dir, int WiDftRecursionLevel, typename T>
__attribute__((always_inline)) inline void cooley_tukey_dft(int factorN, int factorM, const T* in, int stride_in,
                                                            T* out, int stride_out, T* privateScratch) {
  for (int i{0}; i < factorM; ++i) {
    // Do a WI dft of factorN size, reading from in and writing to the private memory.
    wi_dft<Dir, WiDftRecursionLevel>(factorN, in + 2 * i * stride_in, factorM * stride_in,
                                     privateScratch + 2 * i * factorN, 1, privateScratch + 2 * factorN * factorM);
#pragma clang loop unroll(full)
    for (int j{0}; j < factorN; ++j) {
      // Apply twiddles to values in private memory.
      auto re_multiplier = twiddle<T>::Re[factorN * factorM][i * j];
      auto im_multiplier = [&]() {
        if constexpr (Dir == direction::FORWARD) {
          return twiddle<T>::Im[factorN * factorM][i * j];
        }
        return -twiddle<T>::Im[factorN * factorM][i * j];
      }();
      T tmp_val = privateScratch[2 * i * factorN + 2 * j] * re_multiplier -
                  privateScratch[2 * i * factorN + 2 * j + 1] * im_multiplier;
      privateScratch[2 * i * factorN + 2 * j + 1] = privateScratch[2 * i * factorN + 2 * j] * im_multiplier +
                                                    privateScratch[2 * i * factorN + 2 * j + 1] * re_multiplier;
      privateScratch[2 * i * factorN + 2 * j + 0] = tmp_val;
    }
  }
#pragma clang loop unroll(full)
  for (int i{0}; i < factorN; ++i) {
    // Do a WI dft of factor M size, reading from private memory and writing to out.
    wi_dft<Dir, WiDftRecursionLevel>(factorM, privateScratch + 2 * i, factorN, out + 2 * i * stride_out,
                                     factorN * stride_out, privateScratch + 2 * factorN * factorM);
  }
}

/**
 * Factorizes a number into two roughly equal factors.
 * @tparam TIndex Index type
 * @param N the number to factorize
 * @return the smaller of the factors
 */
template <typename TIndex>
constexpr TIndex factorize(TIndex N) {
  TIndex res = 1;
  for (TIndex i = 2; i * i <= N; i++) {
    if (N % i == 0) {
      res = i;
    }
  }
  return res;
}

/**
 * Calculates how many temporary complex values a workitem implementation needs
 * for solving FFT.
 * @tparam TIndex Index type
 * @tparam RecursionDepth How many times has this function called itself alread. Default 0.
 * @param N size of the FFT problem
 * @return Number of temporary complex values
 */
template <typename TIndex, int RecursionDepth = 0>
constexpr TIndex wi_temps(TIndex N) {
  // This function is recursive, but DPC++ can compile it. The depth
  // limit is because its annoying for the compiler warn us about the recursion.
  constexpr int MaxRecursionLevel = 10;  // 2^10 allows dftSize < 2^10 == 1024.
  static_assert((1UL << MaxRecursionLevel) > detail::MaxFftSizeWi,
                "Insufficient max recursion level for maximum allowable DFT size.");
  if constexpr (RecursionDepth < MaxRecursionLevel){
    TIndex f0 = factorize(N);
    TIndex f1 = N / f0;
    if (f0 < 2 || f1 < 2) {
      return N;
    }
    TIndex a = wi_temps<TIndex, RecursionDepth + 1>(f0);
    TIndex b = wi_temps<TIndex, RecursionDepth + 1>(f1);
    return (a > b ? a : b) + N;
  } else {
    return 0;
  }
}

/**
 * Checks whether a problem can be solved with workitem implementation without
 * registers spilling.
 * @tparam Scalar type of the real scalar used for the computation
 * @tparam TIndex Index type
 * @param N Size of the problem, in complex values
 * @return true if the problem fits in the registers
 */
template <typename Scalar, typename TIndex>
constexpr bool fits_in_wi(TIndex N) {
  TIndex n_complex = N + wi_temps(N);
  TIndex complex_size = 2 * sizeof(Scalar);
  TIndex register_space = PORTFFT_REGISTERS_PER_WI * 4;
  return n_complex * complex_size <= register_space;
}

};  // namespace detail

/**
 * Calculates DFT using FFT algorithm. Can work in or out of place.
 *
 * @tparam Dir direction of the FFT
 * @tparam WiDftRecursionLevel How many times has has wi_dft been recursively called before?
 * @tparam T type of the scalar used for computations
 * @param dftSize size of the DFT transform
 * @param in pointer to input
 * @param stride_in stride (in complex values) between complex values in `in`
 * @param out pointer to output
 * @param stride_out stride (in complex values) between complex values in `out`
 * @param privateScratch Scratch memory for this WI.
 */
template <direction Dir, int WiDftRecursionLevel, typename T>
__attribute__((always_inline)) inline void wi_dft(int dftSize, const T* in, int stride_in, T* out, int stride_out,
                                                  T* privateScratch) {
  int f0 = detail::factorize(dftSize);
  constexpr int MaxRecursionLevel = detail::uint_log2(detail::MaxFftSizeWi) - 1;
  if constexpr (WiDftRecursionLevel < MaxRecursionLevel) {
    if (dftSize == 2) {
      T a = in[0 * stride_in + 0] + in[2 * stride_in + 0];
      T b = in[0 * stride_in + 1] + in[2 * stride_in + 1];
      T c = in[0 * stride_in + 0] - in[2 * stride_in + 0];
      out[2 * stride_out + 1] = in[0 * stride_in + 1] - in[2 * stride_in + 1];
      out[0 * stride_out + 0] = a;
      out[0 * stride_out + 1] = b;
      out[2 * stride_out + 0] = c;
    } else if (f0 >= 2 && dftSize / f0 >= 2) {
      detail::cooley_tukey_dft<Dir, WiDftRecursionLevel + 1>(dftSize / f0, f0, in, stride_in, out, stride_out,
                                                             privateScratch);
    } else {
      detail::naive_dft<Dir>(dftSize, in, stride_in, out, stride_out, privateScratch);
    }
  }
}

};  // namespace portfft

#endif
