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

#include <defines.hpp>
#include <enums.hpp>
#include <sycl/sycl.hpp>

namespace portfft::detail {

/**
 * Struct containing objects that are used in almost all functions.
 */
struct global_data_struct {
#ifdef PORTFFT_LOG
  sycl::stream s;
#endif
  sycl::nd_item<1> it;
  sycl::sub_group sg;
  sycl::kernel_handler kh;

  /**
   * Constructor.
   *
   * @param s sycl::stream for logging
   * @param it nd_item of the kernel
   */
  global_data_struct(
#ifdef PORTFFT_LOG
      sycl::stream s,
#endif
      sycl::nd_item<1> it, sycl::kernel_handler kh)
      :
#ifdef PORTFFT_LOG
        s(s << sycl::setprecision(3)),
#endif
        it(it),
        sg(it.get_sub_group()),
        kh(kh) {
  }

  /** Get the group for this work-item associated with a level.
   */
  template <level Level>
  __attribute__((always_inline)) inline auto get_group() {
    static_assert(Level == level::SUBGROUP || Level == level::WORKGROUP,
                  "No group associated with WORKITEM or DEVICE levels");
    if constexpr (Level == level::SUBGROUP) {
      return sg;
    } else {
      return it.get_group();
    }
  }

#ifdef PORTFFT_LOG
  /**
   * Logs ids of workitem, subgroup and workgroup.
   */
  __attribute__((always_inline)) inline void log_ids() const {
    s << "wg_id " << it.get_group(0);
    s << " sg_id_in_wg " << it.get_local_id(0) / sg.get_local_range()[0];
    s << " sg_loc_id " << sg.get_local_linear_id() << " ";
  }

  /**
   * Implementation of log_message. End of recursion - logs the message, adds a newline and flushes the stream.
   *
   * @tparam T type of the object to log
   * @param message message to log
   */
  template <typename T>
  __attribute__((always_inline)) inline void log_message_impl(T message) {
    s << message << "\n" << sycl::stream_manipulator::flush;
  }

  /**
   * Implementation of log_message. End of recursion - logs the messages separated by newlines, adds a newline and
   * flushes the stream.
   *
   * @tparam TFirst type of the first object to log
   * @tparam Ts types of the other objects to log
   * @param message the first message to log
   * @param other_messages other messages to log
   */
  template <typename TFirst, typename... Ts>
  __attribute__((always_inline)) inline void log_message_impl(TFirst message, Ts... other_messages) {
    s << message << " ";
    log_message_impl(other_messages...);
  }
#endif

  /**
   * Logs content of the local memory. Also outputs the id of the workgroup it is called from.
   *
   * Does nothing if logging of dumps is not enabled (PORTFFT_LOG_DUMPS is not defined).
   *
   * @tparam ViewT View type of data to log
   * @param message message to log before the data
   * @param data View of data to log
   * @param num number of elements to log
   */
  template <typename ViewT>
  PORTFFT_INLINE void log_dump_local([[maybe_unused]] const char* message, [[maybe_unused]] ViewT data,
                                     [[maybe_unused]] Idx num) {
#ifdef PORTFFT_LOG_DUMPS
    if (it.get_local_id(0) == 0) {
      s << "wg_id " << it.get_group(0);
      s << " " << message << " ";
      if (num) {
        s << data[0];
      }
      for (Idx i = 1; i < num; i++) {
        s << ", " << data[i];
      }
      s << "\n" << sycl::stream_manipulator::flush;
    }
#endif
  }

  /**
   * Logs content of the private memory. Also outputs the ids of the workitem, subgroup and workgroup it is called from.
   *
   * Does nothing if logging of dumps is not enabled (PORTFFT_LOG_DUMPS is not defined).
   *
   * @tparam T type of the data to log
   * @param message message to log before the data
   * @param ptr pointer to data to log
   * @param num number of elements to log
   */
  template <typename T>
  PORTFFT_INLINE void log_dump_private([[maybe_unused]] const char* message, [[maybe_unused]] T* ptr,
                                       [[maybe_unused]] Idx num) {
#ifdef PORTFFT_LOG_DUMPS
    log_ids();
    s << message << " ";
    if (num) {
      s << ptr[0];
    }
    for (Idx i = 1; i < num; i++) {
      s << ", " << ptr[i];
    }
    s << "\n" << sycl::stream_manipulator::flush;
#endif
  }

  /**
   * Logs a message. Can log multiple objects/strings. They will be separated by spaces.
   *
   * Does nothing if logging of transfers is not enabled (PORTFFT_LOG_TRANSFERS is not defined).
   *
   * @tparam Ts types of the objects to log
   * @param messages objects to log
   */
  template <typename... Ts>
  PORTFFT_INLINE void log_message([[maybe_unused]] Ts... messages) {
#ifdef PORTFFT_LOG_TRANSFERS
    log_ids();
    log_message_impl(messages...);
#endif
  }

  /**
   * Logs a message from a subgroup - there will be only one output from each subgroup. Can log multiple
   * objects/strings. They will be separated by spaces. Also outputs the id of the subgroup and workgroup it is called
   * from.
   *
   * Does nothing if logging of transfers is not enabled (PORTFFT_LOG_TRANSFERS is not defined).
   *
   * @tparam Ts types of the objects to log
   * @param messages objects to log
   */
  template <typename... Ts>
  PORTFFT_INLINE void log_message_subgroup([[maybe_unused]] Ts... messages) {
#ifdef PORTFFT_LOG_TRANSFERS
    if (sg.leader()) {
      s << "sg_id " << sg.get_group_linear_id() << " "
        << "wg_id " << it.get_group(0) << " ";
      log_message_impl(messages...);
    }
#endif
  }

  /**
   * Logs a message from a workgroup - there will be only one output from each workgroup. Can log multiple
   * objects/strings. They will be separated by spaces. Also outputs the id of the workgroup it is called from.
   *
   * Does nothing if logging of transfers is not enabled (PORTFFT_LOG_TRANSFERS is not defined).
   *
   * @tparam Ts types of the objects to log
   * @param messages objects to log
   */
  template <typename... Ts>
  PORTFFT_INLINE void log_message_local([[maybe_unused]] Ts... messages) {
#ifdef PORTFFT_LOG_TRANSFERS
    if (it.get_local_id(0) == 0) {
      s << "wg_id " << it.get_group(0) << " ";
      log_message_impl(messages...);
    }
#endif
  }

  /**
   * Logs a message from the kernel - there will be only one output from the kernel. Can log multiple objects/strings.
   * They will be separated by spaces.
   *
   * Does nothing if logging of transfers is not enabled (PORTFFT_LOG_TRACE is not defined).
   *
   * @tparam Ts types of the objects to log
   * @param messages objects to log
   */
  template <typename... Ts>
  PORTFFT_INLINE void log_message_global([[maybe_unused]] Ts... messages) {
#ifdef PORTFFT_LOG_TRACE
    if (it.get_global_id(0) == 0) {
      log_message_impl(messages...);
    }
#endif
  }

  /**
   * Logs a message with a single message from the selected level. Can log multiple objects/strings. They will be
   * separated by spaces. Also outputs info on the calling level. Does not support DEVICE level.
   *
   * Does nothing if logging of transfers is not enabled (PORTFFT_LOG_TRANSFERS is not defined).
   *
   * @tparam Level The level of granularity with which to output logging data
   * @tparam Ts types of the objects to log
   * @param messages objects to log
   */
  template <level Level, typename... Ts>
  PORTFFT_INLINE void log_message_scoped([[maybe_unused]] Ts... messages) {
    static_assert(Level == level::WORKITEM || Level == level::SUBGROUP || Level == level::WORKGROUP,
                  "Only WORKITEM, SUBGROUP and WORKGROUP levels are supported");
    if constexpr (Level == level::WORKITEM) {
      log_message(messages...);
    } else if constexpr (Level == level::SUBGROUP) {
      log_message_subgroup(messages...);
    } else if constexpr (Level == level::WORKGROUP) {
      log_message_local(messages...);
    }
    // DEVICE is not supported because log_message_global uses PORTFFT_LOG_TRACE instead of PORTFFT_LOG_TRANSFERS.
  }
};

};  // namespace portfft::detail

#endif
