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

#include "portfft/common/compiletime_tuning_profile.hpp"
#include "portfft/defines.hpp"

#include <unordered_map>

#ifndef PORTFFT_COMMON_RUNTIME_TUNING_PROFILE_HPP
#define PORTFFT_COMMON_RUNTIME_TUNING_PROFILE_HPP

namespace portfft::detail {

// Built-in runtime-tunable parameter selector
enum class rt_config {
  nvidia,
  nvidia_a100 = nvidia,
  amd,
  amd_mi210 = amd,
  amd_w6800 = amd,
  intel,
  intel_pvc,
  intel_arc = intel,
  intel_uhd = intel
};

// A class to represent runtime tunable parameters
struct rt_config_desc {
  Idx sgs_per_wg;
  ct_profile preferred_ct_profile;
};

static std::unordered_map<rt_config, rt_config_desc> rt_config_descs{
    {rt_config::nvidia, {4, ct_profile::nvidia}},
    {rt_config::amd, {8, ct_profile::amd}},
    {rt_config::intel, {2, ct_profile::intel_arc}},
    {rt_config::intel_pvc, {4, ct_profile::intel_pvc}}};

}  // namespace portfft::detail

#endif  // PORTFFT_COMMON_RUNTIME_TUNING_PROFILE_HPP
