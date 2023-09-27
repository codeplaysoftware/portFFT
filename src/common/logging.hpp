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

#ifndef PORTFFT_COMMON_LOGGING_HPP
#define PORTFFT_COMMON_LOGGING_HPP

#include <sycl/sycl.hpp>

namespace portfft::detail {

struct global_data_struct{
#ifdef PORTFFT_LOG
  sycl::stream s;
#endif
  sycl::nd_item<1> it;
  sycl::sub_group sg;

#ifdef PORTFFT_LOG
  __attribute__((always_inline)) inline void log_ids() const {
    s << "wg_id " << it.get_group(0);
    s << " sg_id_in_wg " << it.get_local_id(0) / sg.get_local_range()[0];
    s << " sg_loc_id " << sg.get_local_linear_id() << " ";
  }

  
  template<typename T>
  __attribute__((always_inline)) inline void log_message_impl(T message){
    s << message << "\n" << sycl::stream_manipulator::flush;
  }

template<typename TFirst, typename... Ts>
  __attribute__((always_inline)) inline void log_message_impl(TFirst message, Ts... other_messages){
    s << message << " ";
    log_message_impl(other_messages...);
  }
#endif

  template<typename T>
  __attribute__((always_inline)) inline void log_dump_local([[maybe_unused]] const char* message, [[maybe_unused]] T* ptr, [[maybe_unused]] std::size_t num){
#ifdef PORTFFT_LOG_DUMPS
    if(it.get_local_id(0)==0){
      s << "wg_id " << it.get_group(0);
      s << " " << message << " ";
      if(num){
        s << ptr[0];
      }
      for(std::size_t i=1;i<num;i++){
        s << ", " << ptr[i];
      }
      s << "\n" << sycl::stream_manipulator::flush;
    }
#endif
  }

    template<typename T>
  __attribute__((always_inline)) inline void log_dump_private([[maybe_unused]] const char* message, [[maybe_unused]] T* ptr, [[maybe_unused]] std::size_t num){
#ifdef PORTFFT_LOG_DUMPS
    log_ids();
    s << message << " ";
    if(num){
      s << ptr[0];
    }
    for(std::size_t i=1;i<num;i++){
      s << ", " << ptr[i];
    }
    s << "\n" << sycl::stream_manipulator::flush;
#endif
  }

  template<typename... Ts>
  __attribute__((always_inline)) inline void log_message([[maybe_unused]] Ts... messages){
#ifdef PORTFFT_LOG_TRANSFERS
    log_ids();
    log_message_impl(messages...);
#endif
  }
  
  template<typename... Ts>
  __attribute__((always_inline)) inline void log_message_local([[maybe_unused]] Ts... messages){
#ifdef PORTFFT_LOG_TRANSFERS
    if(it.get_local_id(0) == 0){
      s << "wg_id " << it.get_group(0) << " ";
      log_message_impl(messages...);
    }
#endif
  }
  
  template<typename... Ts>
  __attribute__((always_inline)) inline void log_message_global([[maybe_unused]] Ts... messages){
#ifdef PORTFFT_LOG_TRANSFERS
    if(it.get_global_id(0) == 0){
      log_message_impl(messages...);
    }
#endif
  }
};



};  // namespace portfft::detail

#endif
