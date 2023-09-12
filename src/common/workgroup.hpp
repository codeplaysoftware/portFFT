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

#ifndef PORTFFT_COMMON_WORKGROUP_HPP
#define PORTFFT_COMMON_WORKGROUP_HPP

#include <common/helpers.hpp>
#include <common/subgroup.hpp>
#include <enums.hpp>

namespace portfft {

/**
 * Calculate the number of groups or bank lines of PORTFFT_N_LOCAL_BANKS between each padding in local memory,
 * specifically for reducing bank conflicts when reading values from the columns of a 2D data layout. e.g. If there are
 * 64 complex elements in a row, then the consecutive values in the same column are 128 floats apart. There are 32
 * banks, each the size of a float, so we only want a padding float every 128/32=4 bank lines to read along the column
 * without bank conflicts.
 *
 * @param row_size the size in bytes of the row. 32 std::complex<float> values would probably have a size of 256 bytes.
 * @return constexpr std::size_t the number of groups of PORTFFT_N_LOCAL_BANKS between each padding in local memory.
 */
constexpr std::size_t bank_lines_per_pad_wg(std::size_t row_size) {
  constexpr std::size_t BankLineSize = sizeof(float) * PORTFFT_N_LOCAL_BANKS;
  if (row_size % BankLineSize == 0) {
    return row_size / BankLineSize;
  }
  // There is room for improvement here. E.G if row_size was half of BankLineSize then maybe you would still want 1
  // pad every bank group.
  return 0;
}

//dfts in one dir
// StrideWithinDFT = number of problems in inner stride dimension
// BetweenProblemOuterStride = DFTSize * StrideWithinDFT
//template<int StrideWithinDFT, int BetweenProblemInnerStride=1, int BetweenProblemOuterStride, int SubgroupSize>
template<direction Dir, int DFTSize, int StrideWithinDFT, int NDFTsInOuterDimension, int SubgroupSize, std::size_t BankLinesPerPad, typename T>
__attribute__((always_inline)) inline void dimension_dft(T* loc, T* priv, T* loc_twiddles, const T* wg_twiddles, T scaling_factor, sycl::nd_item<1> it, sycl::stream s){
  constexpr int OuterStride = DFTSize * StrideWithinDFT;
  // the number of work-items involved in every subgroup fft
  constexpr int FactSg = detail::factorize_sg(DFTSize, SubgroupSize);
  // the number of values held in by a work-item in a row subgroup dft
  constexpr int FactWi = DFTSize / FactSg;

  constexpr int FFTsPerSG = SubgroupSize / FactSg;
  constexpr bool ExcessWIs = SubgroupSize % FactSg > 0;
  constexpr bool ExcessSGs = StrideWithinDFT % FFTsPerSG > 0;
  // only needed when there are excess work-items
  constexpr std::size_t MaxWorkingTidInSg = FFTsPerSG * FactSg;

  const int num_sgs = static_cast<int>(it.get_local_range(0)) / SubgroupSize;
  sycl::sub_group sg = it.get_sub_group();
  const int fft_in_subgroup = static_cast<int>(sg.get_local_linear_id()) / FactSg;
  // id of the work-item in the fft
  const int wi_id_in_fft = static_cast<int>(sg.get_local_linear_id()) % FactSg;

  const int begin = static_cast<int>(sg.get_group_id()) * FFTsPerSG + fft_in_subgroup;
  const int step = num_sgs * FFTsPerSG;
  int end;
  constexpr int TotalDFTs = StrideWithinDFT * NDFTsInOuterDimension;
  if constexpr (ExcessSGs) {
    // sg_dft uses subgroup operations, so all of the subgroup must enter the loop
    // it is safe to increase column_end for all work-items since they are all taking steps of FFTsPerSG anyway
    end = detail::round_up_to_multiple(TotalDFTs, FFTsPerSG);
  } else {
    end = TotalDFTs;
  }

  if constexpr (ExcessWIs) {
    // also allow these work-items to enter the loop, without making other work-items do another loop.
    end += (fft_in_subgroup == FFTsPerSG) ? 1 : 0;
  }
  sycl::group_barrier(it.get_group());
  //if(sg.get_local_linear_id()==0) s << sg.get_group_id() << "START of DIM - loop bounds " << begin << " " << end << " " << step << "\n";
  //if(sg.get_local_linear_id()==0) s << "DFTSize " << DFTSize << " StrideWithinDFT " << StrideWithinDFT << " NDFTsInOuterDimension " << NDFTsInOuterDimension << "\n";
  //if(sg.get_local_linear_id()==0) s << "FactWi " << FactWi << " FactSg " << FactSg << "\n" << sycl::stream_manipulator::flush;
  //sycl::group_barrier(it.get_group());

  //for(int j = 0; j < NDFTsInOuterDimension; j++){
    //T* loc_start = loc + j * OuterStride;
    for (int j = begin; j < end; j += step) {
      int j_inner = j % StrideWithinDFT;
      int j_outer = j / StrideWithinDFT;
      T* loc_start = loc + detail::pad_local(2 * j_outer * OuterStride, BankLinesPerPad);
      bool working = true;
      if constexpr (ExcessSGs) {
        working = j < TotalDFTs;
      }
      if constexpr (ExcessWIs) {
        working = working && sg.get_local_linear_id() < MaxWorkingTidInSg;
      }
      if (working) {
        /*if(it.get_global_id(0)==0){
          s << "local2private_transposed new " 
            << FactWi << " "
            << BankLinesPerPad << " "
            << wi_id_in_fft << " "
            << j_inner << " "
            << StrideWithinDFT << "\n" << sycl::stream_manipulator::flush;
        }*/
        //transposition due to working on columns
        local2private_transposed<FactWi, detail::pad::DO_PAD, BankLinesPerPad>(loc_start, priv, wi_id_in_fft, j_inner, StrideWithinDFT);
          sycl::group_barrier(it.get_group());
        /*if(it.get_global_id(0)==0){
          s << "after load\n\n" << sycl::stream_manipulator::flush;
        }*/
        //sycl::group_barrier(it.get_group());
        /*for(int k=0;k<FactSg;k++){
          sycl::group_barrier(it.get_group());
          if(it.get_local_linear_id()==k && j <= begin+step){
            s  << k << ": ";
            for(int l=0;l<FactWi*2;l++){
              s << priv[l] << ", ";
            }
            s << "\n" << sycl::stream_manipulator::flush;
          }
        }*/
        //if(j==begin+step) s << it.get_global_id(0) << " " << StrideWithinDFT << ": " << priv[1] << "\n";
        if(wg_twiddles){
          detail::unrolled_loop<0, FactWi, 1>([&](const int i) __attribute__((always_inline)) {
            int twiddle_i = i + wi_id_in_fft * FactWi;
            int twiddle_j = j_outer;
            //s << it.get_local_linear_id() << " twiddle: " << twiddle_i << " " << twiddle_j << "\n" << sycl::stream_manipulator::flush;
            /*auto tw = detail::calculate_twiddle<T>(twiddle_i * twiddle_j, DFTSize * NDFTsInOuterDimension);
            T twiddle_real = tw.real();
            T twiddle_imag = tw.imag();*/
            // TODO coalesced?
            int twiddle_index = twiddle_i * NDFTsInOuterDimension + twiddle_j;
            sycl::vec<T, 2> twiddles = reinterpret_cast<const sycl::vec<T, 2>*>(wg_twiddles)[twiddle_index];
            T twiddle_real = twiddles[0];
            T twiddle_imag = twiddles[1];

            /*int element = 2*(wi_id_in_fft * FactWi + i); //2 * i * FactSg + 2 * wi_id_in_fft; //wi_id_in_fft * FactWi + i;
            int twiddle_index = 2 * NDFTsInOuterDimension * j_outer + element; //DFTSize * i + element;
            s << it.get_local_linear_id() << " twiddle: " << twiddle_index << "\n" << sycl::stream_manipulator::flush;
            sycl::vec<T, 2> twiddles = *reinterpret_cast<const sycl::vec<T, 2>*>(&wg_twiddles[twiddle_index]);
            T twiddle_real = twiddles[0];
            T twiddle_imag = twiddles[1];*/
            if constexpr (Dir == direction::BACKWARD) {
              twiddle_imag = -twiddle_imag;
            }
            T tmp_real = priv[2 * i];
            priv[2 * i] = tmp_real * twiddle_real - priv[2 * i + 1] * twiddle_imag;
            priv[2 * i + 1] = tmp_real * twiddle_imag + priv[2 * i + 1] * twiddle_real;
          });
          /*if (working) {
            if(it.get_global_id(0)==0){
              s << "after twiddling\n\n" << sycl::stream_manipulator::flush;
            }
            for(int k=0;k<FactSg;k++){
              sycl::group_barrier(it.get_group());
              if(it.get_local_linear_id()==k && j <= begin+step){
                s  << k << ": ";
                for(int l=0;l<FactWi*2;l++){
                  s << priv[l] << ", ";
                }
                s << "\n" << sycl::stream_manipulator::flush;
              }
            }
          }*/
        }
        if(scaling_factor != T(1)){
          detail::unrolled_loop<0, FactWi, 1>([&](const int i) __attribute__((always_inline)) {
            priv[2 * i] *= scaling_factor;
            priv[2 * i + 1] *= scaling_factor;
          });
        }
      }
      //s << "sg_dft " << FactWi << ", " << FactSg << ", 0" << "\n" << sycl::stream_manipulator::flush;
      sg_dft<Dir, FactWi, FactSg>(priv, sg, loc_twiddles);
      if (working) {
        //transposition due to working on columns AND transposition for SG dft
        
        /*if(it.get_global_id(0)==0){
          s << "after compute\n\n" << sycl::stream_manipulator::flush;
        }
        for(int k=0;k<FactSg;k++){
          sycl::group_barrier(it.get_group());
          if(it.get_local_linear_id()==k && j <= begin+step){
            s  << k << ": ";
            for(int l=0;l<FactWi*2;l++){
              s << priv[l] << ", ";
            }
            s << "\n" << sycl::stream_manipulator::flush;
          }
        }*/
        //private2local_transposed<FactWi, detail::pad::DO_PAD, BankLinesPerPad>(priv, loc_start, wi_id_in_fft, FactSg,
          //                                                                      j_inner, StrideWithinDFT);
        /*private2local_2strides<FactWi, detail::pad::DO_PAD, BankLinesPerPad>(priv, loc, wi_id_in_fft, 
                                                                              FactSg*StrideWithinDFT,
                                                                              j_inner + 2 * j_outer * OuterStride, 
                                                                              StrideWithinDFT, s);*/
        private2local_2strides<FactWi, detail::pad::DO_PAD, BankLinesPerPad>(priv, loc, wi_id_in_fft, 
                                                                              FactSg*StrideWithinDFT,
                                                                              j_inner + j_outer * OuterStride, 
                                                                              StrideWithinDFT, s);
      }
    }
  //}
}

/**
 * Calculates FFT using Bailey 4 step algorithm.
 *
 * @tparam Dir Direction of the FFT
 * @tparam FFTSize Problem Size
 * @tparam N Smaller factor of the Problem size
 * @tparam M Larger factor of the problem size
 * @tparam SubgroupSize Size of the subgroup
 * @tparam BankLinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T Scalar Type
 *
 * @param loc local accessor containing the input
 * @param loc_twiddles Pointer to twiddles to be used by sub group FFTs
 * @param wg_twiddles Pointer to precalculated twiddles which are to be used before second set of FFTs
 * @param it Associated nd_item
 * @param scaling_factor Scalar value with which the result is to be scaled
 */
template <direction Dir, int FFTSize, int N, int M, int SubgroupSize, std::size_t BankLinesPerPad, typename T>
__attribute__((always_inline)) inline void wg_dft(T* loc, T* loc_twiddles, const T* wg_twiddles, sycl::nd_item<1> it,
                                                  T scaling_factor, sycl::stream s) {
  // the number of work-items involved in every row subgroup fft
  constexpr int FactSgN = detail::factorize_sg(N, SubgroupSize);
  // the number of values held in by a work-item in a row subgroup dft
  constexpr int FactWiN = N / FactSgN;
  // the number of work-items involved in every column subgroup fft
  constexpr int FactSgM = detail::factorize_sg(M, SubgroupSize);
  // the number of values held in by a work-item in a column subgroup dft
  constexpr int FactWiM = M / FactSgM;

  constexpr int PrivateMemSize = FactWiM > FactWiN ? 2 * FactWiM : 2 * FactWiN;
  T priv[PrivateMemSize];
  const int num_sgs = static_cast<int>(it.get_local_range(0)) / SubgroupSize;
  sycl::sub_group sg = it.get_sub_group();

  sycl::group_barrier(it.get_group());
  /*if(it.get_local_linear_id()==0){
    s << "factors FactSgN " << FactSgN << " FactWiN " << FactWiN << " FactSgM " << FactSgM << " FactWiM " << FactWiM << "\n"; 
    s << "first dim " << FFTSize << " N " << N << " M " << M << "\n"; 
    s << "\n\n\n";
    for(int i=0;i<M*N;i++){
      s << loc[i] << ",";
    }
    s << "\n\n\n";
  }*/
  sycl::group_barrier(it.get_group());
  dimension_dft<Dir, N, M, 1, SubgroupSize, BankLinesPerPad, T>(loc, priv, loc_twiddles + (2 * M), nullptr, 1, it, s);
  /*{  // column ffts
    constexpr int FFTsPerSG = SubgroupSize / FactSgN;
    constexpr bool ExcessWIs = SubgroupSize % FactSgN > 0;
    constexpr bool ExcessSGs = M % FFTsPerSG > 0;

    // only needed when there are excess work-items
    constexpr std::size_t MaxWorkingTidInSg = FFTsPerSG * FactSgN;

    const int fft_in_subgroup = static_cast<int>(sg.get_local_linear_id()) / FactSgN;
    // id of the work-item in the fft
    const int wi_id_in_fft = static_cast<int>(sg.get_local_linear_id()) % FactSgN;

    const int column_begin = static_cast<int>(sg.get_group_id()) * FFTsPerSG + fft_in_subgroup;
    const int column_step = num_sgs * FFTsPerSG;
    int column_end;
    if constexpr (ExcessSGs) {
      // sg_dft uses subgroup operations, so all of the subgroup must enter the loop
      // it is safe to increase column_end for all work-items since they are all taking steps of FFTsPerSG anyway
      column_end = detail::round_up_to_multiple(M, FFTsPerSG);
    } else {
      column_end = M;
    }

    if constexpr (ExcessWIs) {
      // also allow these work-items to enter the loop, without making other work-items do another loop.
      column_end += (fft_in_subgroup == FFTsPerSG) ? 1 : 0;
    }

    for (int column = column_begin; column < column_end; column += column_step) {
      bool working = true;
      if constexpr (ExcessSGs) {
        working = column < M;
      }
      if constexpr (ExcessWIs) {
        working = working && sg.get_local_linear_id() < MaxWorkingTidInSg;
      }
      if (working) {
        local2private_transposed<FactWiN, detail::pad::DO_PAD, BankLinesPerPad>(loc, priv, wi_id_in_fft, column, M, s, it);
        if(column <= column_begin+column_step){
          if(it.get_global_id(0)==0){
            s << "after load\n\n" << sycl::stream_manipulator::flush;
          }
          sycl::group_barrier(it.get_group());
          for(int k=0;k<FactSgN;k++){
            sycl::group_barrier(it.get_group());
            if(it.get_local_linear_id()==k){
              s  << k << ": ";
              for(int l=0;l<FactWiN*2;l++){
                s << priv[l] << ", ";
              }
              s << "\n" << sycl::stream_manipulator::flush;
            }
          }
        }
      }
      sg_dft<Dir, FactWiN, FactSgN>(priv, sg, loc_twiddles + (2 * M));
      if (working) {

        //private2local_transposed<FactWiN, detail::pad::DO_PAD, BankLinesPerPad>(priv, loc, wi_id_in_fft, FactSgN,
          //                                                                      column, M);
        private2local_2strides<FactWiN, detail::pad::DO_PAD, BankLinesPerPad>(priv, loc, wi_id_in_fft, FactSgN*M,
                                                                                column, M);
        //private2local_2strides<FactWiM, detail::pad::DO_PAD, BankLinesPerPad>(priv, loc, wi_id_in_fft, FactSgN*M, 
            //column, M);
      }
    }
  }//*/
  sycl::group_barrier(it.get_group());
  /*if(it.get_local_linear_id()==0){
    s << "second dim " << FFTSize << " N " << N << " M " << M << "\n"; 
    s << "\n\n\n";
    for(int i=0;i<M*N;i++){
      s << loc[i] << ",";
    }
    s << "\n\n\n";
  }*/
  
  sycl::group_barrier(it.get_group());
  dimension_dft<Dir, M, 1, N, SubgroupSize, BankLinesPerPad, T>(loc, priv, loc_twiddles, wg_twiddles, scaling_factor, it, s);
  /*{  // row ffts
    constexpr int FFTsPerSG = SubgroupSize / FactSgM;
    constexpr bool ExcessWIs = SubgroupSize % FactSgM > 0;
    constexpr bool ExcessSGs = N % FFTsPerSG > 0;

    // only needed when there are excess work-items
    constexpr int MaxWorkingTidInSg = FFTsPerSG * FactSgM;

    const int fft_in_subgroup = static_cast<int>(sg.get_local_linear_id()) / FactSgM;
    // id of the work-item in the fft
    const int wi_id_in_fft = static_cast<int>(sg.get_local_linear_id()) % FactSgM;

    const int row_begin = static_cast<int>(sg.get_group_id()) * FFTsPerSG + fft_in_subgroup;
    const int row_step = num_sgs * FFTsPerSG;
    int row_end;
    if constexpr (ExcessSGs) {
      // sg_dft uses subgroup operations, so all of the subgroup must enter the loop
      // it is safe to increase column_end for all work-items since they are all taking steps of FFTsPerSG anyway
      row_end = detail::round_up_to_multiple(N, FFTsPerSG);
    } else {
      row_end = N;
    }
    //if(it.get_global_id(0)==0) s << "START of old - loop bounds" << row_begin << " " << row_end << " " << row_step << "\n";
    //if(it.get_global_id(0)==0) s << "ExcessWIs " << ExcessWIs << " ExcessSGs " << ExcessSGs << "\n";

    if constexpr (ExcessWIs) {
      // also allow these work-items to enter the loop, without making other work-items do another loop.
      row_end += (fft_in_subgroup == FFTsPerSG) ? 1 : 0;
    }

    for (int row = row_begin; row < row_end; row += row_step) {
      bool working = true;
      if constexpr (ExcessSGs) {
        working = row < N;
      }
      if constexpr (ExcessWIs) {
        working = working && sg.get_local_linear_id() < MaxWorkingTidInSg;
      }
      if (working) {
        if(it.get_global_id(0)==0){
          s << "local2private og " 
            << 2 * FactWiM << " "
            << BankLinesPerPad << " "
            << wi_id_in_fft << " "
            << 2 * FactWiM << " "
            << 2 * M * row << "\n" << sycl::stream_manipulator::flush;
        }
        local2private<2 * FactWiM, detail::pad::DO_PAD, BankLinesPerPad>(
            loc, priv, static_cast<std::size_t>(wi_id_in_fft), static_cast<std::size_t>(2 * FactWiM),
            static_cast<std::size_t>(2 * M * row));
        if(it.get_global_id(0)==0){
          s << "after load\n\n" << sycl::stream_manipulator::flush;
        }
        sycl::group_barrier(it.get_group());
        for(int k=0;k<FactSgM;k++){
          sycl::group_barrier(it.get_group());
          if(it.get_local_linear_id()==k && row <= row_begin+row_step){
            s  << k << ": ";
            for(int l=0;l<FactWiM*2;l++){
              s << priv[l] << ", ";
            }
            s << "\n" << sycl::stream_manipulator::flush;
          }
        }
      }
      detail::unrolled_loop<0, FactWiM, 1>([&](const int i) __attribute__((always_inline)) {
        int element = 2 * i * FactSgM + 2 * wi_id_in_fft;
        int twiddle_index = 2 * M * row + element;
        sycl::vec<T, 2> twiddles = *reinterpret_cast<const sycl::vec<T, 2>*>(&wg_twiddles[twiddle_index]);
        T twiddle_real = twiddles[0];
        T twiddle_imag = twiddles[1];
        if constexpr (Dir == direction::BACKWARD) {
          twiddle_imag = -twiddle_imag;
        }
        T tmp_real = priv[2 * i];
        priv[2 * i] = tmp_real * twiddle_real - priv[2 * i + 1] * twiddle_imag;
        priv[2 * i + 1] = tmp_real * twiddle_imag + priv[2 * i + 1] * twiddle_real;
      });
      if(working && row <= row_begin+row_step){
        if(it.get_global_id(0)==0){
          s << "after twiddles\n\n" << sycl::stream_manipulator::flush;
        }
        sycl::group_barrier(it.get_group());
        for(int k=0;k<FactSgM;k++){
          sycl::group_barrier(it.get_group());
          if(it.get_local_linear_id()==k){
            s  << k << ": ";
            for(int l=0;l<FactWiM*2;l++){
              s << priv[l] << ", ";
            }
            s << "\n" << sycl::stream_manipulator::flush;
          }
        }
      }
      
      s << "sg_dft " << FactWiM << ", " << FactSgM << ", 0" << "\n" << sycl::stream_manipulator::flush;
      sg_dft<Dir, FactWiM, FactSgM>(priv, sg, loc_twiddles);
      detail::unrolled_loop<0, FactWiM, 1>([&](const int i) __attribute__((always_inline)) {
        priv[2 * i] *= scaling_factor;
        priv[2 * i + 1] *= scaling_factor;
      });

      if (working) {
        if(row <= row_begin+row_step){
          if(it.get_global_id(0)==0){
            s << "after compute\n\n" << sycl::stream_manipulator::flush;
          }
          sycl::group_barrier(it.get_group());
          for(int k=0;k<FactSgM;k++){
            sycl::group_barrier(it.get_group());
            if(it.get_local_linear_id()==k){
              s  << k << ": ";
              for(int l=0;l<FactWiM*2;l++){
                s << priv[l] << ", ";
              }
              s << "\n" << sycl::stream_manipulator::flush;
            }
          }
        }
        //store_transposed<2 * FactWiM, detail::pad::DO_PAD, BankLinesPerPad>(
          //  priv, loc, static_cast<std::size_t>(wi_id_in_fft), static_cast<std::size_t>(FactSgM),
            //static_cast<std::size_t>(2 * M * row));
        private2local_2strides<FactWiM, detail::pad::DO_PAD, BankLinesPerPad>(
            priv, loc, static_cast<std::size_t>(wi_id_in_fft), static_cast<std::size_t>(FactSgM), 
            static_cast<std::size_t>(M * row), 1);
      }
    }
  }//*/
  
  sycl::group_barrier(it.get_group());
  /*if(it.get_local_linear_id()==0){
    s << "end " << FFTSize << " N " << N << " M " << M << "\n"; 
    s << "\n\n\n";
    for(int i=0;i<M*N;i++){
      s << loc[i] << ",";
    }
    s << "\n\n\n";
  }*/
}

}  // namespace portfft

#endif
