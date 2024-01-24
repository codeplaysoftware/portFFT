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

#include "portfft/defines.hpp"
#include <string_view>

#ifndef PORTFFT_COMMON_COMPILETIME_TUNING_PROFILE_HPP
#define PORTFFT_COMMON_COMPILETIME_TUNING_PROFILE_HPP

namespace portfft::detail {

// Enum to choose compile-time profile
enum class ct_profile { nvidia, amd, intel_pvc, intel_arc, custom_defined, invalid };

/** Class to represent compile-time constants used in SYCL kernel compilation.
 *  Each ct_profile corresponds to a single kernel_spec<ct_profile> defintion.
 *  Each profile adds to binary size and compilation size.
 */
template <ct_profile>
struct kernel_spec {
  static constexpr bool ProfileEnabled = false;
};

#ifdef PORTFFT_BUILD_NVIDIA_KERNEL_PROFILE
template <>
struct kernel_spec<ct_profile::nvidia> {
  static constexpr bool ProfileEnabled = true;
  static constexpr std::string_view Name = "Nvidia";
  static constexpr Idx SgSize = 32;
  static constexpr Idx VecLoadBytes = 16;
  static constexpr bool UseSgTransfers = false;
};
#endif

#ifdef PORTFFT_BUILD_AMD_KERNEL_PROFILE
template <>
struct kernel_spec<ct_profile::amd> {
  static constexpr bool ProfileEnabled = true;
  static constexpr std::string_view Name = "AMD";
  static constexpr Idx SgSize = 32;
  static constexpr Idx VecLoadBytes = 8;
  static constexpr bool UseSgTransfers = false;
};
#endif

#ifdef PORTFFT_BUILD_INTEL_KERNEL_PROFILE
template <>
struct kernel_spec<ct_profile::intel_pvc> {
  static constexpr bool ProfileEnabled = true;
  static constexpr std::string_view Name = "IntelPVC";
  static constexpr Idx SgSize = 16;
  static constexpr Idx VecLoadBytes = 32;
  static constexpr bool UseSgTransfers = true;
};

template <>
struct kernel_spec<ct_profile::intel_arc> {
  static constexpr bool ProfileEnabled = true;
  static constexpr std::string_view Name = "IntelARC";
  static constexpr Idx SgSize = 8;
  static constexpr Idx VecLoadBytes = 32;
  static constexpr bool UseSgTransfers = true;
};
#endif  // PORTFFT_BUILD_INTEL_KERNEL_PROFILE

#ifdef PORTFFT_BUILD_CUSTOM_KERNEL_PROFILE
template <>
struct kernel_spec<ct_profile::custom_defined> {
  static constexpr bool ProfileEnabled = true;
  static constexpr std::string_view Name = "Custom";
  static constexpr Idx SgSize = PORTFFT_BUILD_CUSTOM_KERNEL_PROFILE_SG_SIZE;
  static constexpr Idx VecLoadBytes = PORTFFT_BUILD_CUSTOM_KERNEL_PROFILE_VEC_LOAD_BYTES;
  static constexpr bool UseSgTransfers = PORTFFT_BUILD_CUSTOM_KERNEL_PROFILE_USE_SG_TRANSFERS;
};
#endif  // PORTFFT_BUILD_CUSTOM_KERNEL_PROFILE

/** How many ElemTs should be loaded at once for best performance?
 * @tparam ElemT Element type to load
 * @tparam KernelSpecT The kernel specification
 * @returns The count of ElemT to load
 */
template <typename ElemT, typename KernelSpecT>
PORTFFT_INLINE constexpr Idx vec_load_elements() {
  Idx res = KernelSpecT::VecLoadBytes / sizeof(ElemT);
  return res > 0 ? res : 1;
}

/** A recursive type to enable iteration over the kernel specifications.
 */
template <ct_profile... Configs>
struct kernel_spec_definitions;

template <ct_profile Config, ct_profile... OtherConfigs>
struct kernel_spec_definitions<Config, OtherConfigs...> {
  static constexpr ct_profile Current = Config;
  static constexpr bool LastProfile = false;
  using next = kernel_spec_definitions<OtherConfigs...>;
};

template <ct_profile Config>
struct kernel_spec_definitions<Config> {
  static constexpr ct_profile Current = Config;
  static constexpr bool LastProfile = true;
};

using ct_kernel_specs = kernel_spec_definitions<ct_profile::nvidia, ct_profile::amd, ct_profile::intel_pvc,
                                                ct_profile::intel_arc, ct_profile::custom_defined>;

namespace ct_profile_traits {

#define MAKE_TRAIT(return_type, name, value)    \
  template <ct_profile Profile>                 \
  struct name {                                 \
    static constexpr return_type Value = value; \
  };

MAKE_TRAIT(bool, get_profile_enabled, kernel_spec<Profile>::ProfileEnabled)
MAKE_TRAIT(Idx, get_sg_size, kernel_spec<Profile>::SgSize)
MAKE_TRAIT(bool, get_use_sg_transfers, kernel_spec<Profile>::UseSgTransfers)
MAKE_TRAIT(Idx, get_vec_load_bytes, kernel_spec<Profile>::VecLoadBytes)
MAKE_TRAIT(std::string_view, get_name, kernel_spec<Profile>::Name)

template <ct_profile Profile, typename RealT>
struct get_vector_load_type {
  static constexpr std::size_t VecSize = std::max(std::size_t(1), kernel_spec<Profile>::VecLoadBytes / sizeof(RealT));
  using Type = sycl::vec<RealT, VecSize>;
};

template <ct_profile Profile, typename RealT>
struct get_alignof_vector_load_type {
  static constexpr std::size_t Value = alignof(typename get_vector_load_type<Profile, RealT>::Type);
};

#undef MAKE_TRAIT

template <typename KernelSpecs, template <ct_profile, typename...> class CtTrait, typename RetType,
          typename... OtherParams>
constexpr RetType apply_ct_trait(ct_profile prof, const RetType& default_return_val) noexcept {
  if (KernelSpecs::Current == prof) {
    if constexpr (kernel_spec<KernelSpecs::Current>::ProfileEnabled) {
      return CtTrait<KernelSpecs::Current, OtherParams...>::Value;
    } else {
      return default_return_val;
    }
  } else {
    if constexpr (!KernelSpecs::LastProfile) {
      return apply_ct_trait<typename KernelSpecs::next, CtTrait, RetType, OtherParams...>(prof, default_return_val);
    } else {
      return default_return_val;
    }
  }
}
}  // namespace ct_profile_traits

// Is a profile enabled?
constexpr bool is_enabled(ct_profile prof) {
  return ct_profile_traits::apply_ct_trait<ct_kernel_specs, ct_profile_traits::get_profile_enabled, bool>(prof, false);
}

// Get the subgroup size for a ct profile
constexpr Idx sg_size(ct_profile prof) {
  return ct_profile_traits::apply_ct_trait<ct_kernel_specs, ct_profile_traits::get_sg_size, Idx>(prof, 0);
}

// Get the name for a ct profile
constexpr bool get_uses_sg_transfers(ct_profile prof) {
  return ct_profile_traits::apply_ct_trait<ct_kernel_specs, ct_profile_traits::get_use_sg_transfers, bool>(prof, false);
}

// Get the vec_load_bytes for a ct profile
constexpr Idx get_vec_load_bytes(ct_profile prof) {
  return ct_profile_traits::apply_ct_trait<ct_kernel_specs, ct_profile_traits::get_vec_load_bytes, Idx>(prof, 0);
}

// Get the alignment of a vec<Scalar, vec_load_bytes/sizeof(Scalar)>
template <typename RealT>
constexpr std::size_t get_align_of_vec_t(ct_profile prof) {
  return ct_profile_traits::apply_ct_trait<ct_kernel_specs, ct_profile_traits::get_alignof_vector_load_type,
                                           std::size_t, RealT>(prof, 0);
}

// Get the name for a ct profile
constexpr std::string_view profile_name(ct_profile prof) {
  constexpr std::string_view DefaultResult = "default";
  return ct_profile_traits::apply_ct_trait<ct_kernel_specs, ct_profile_traits::get_name, std::string_view>(
      prof, DefaultResult);
}

template <typename Fn, typename KernelSpecs = ct_kernel_specs>
constexpr ct_profile find_ct_profile_with_predicate(Fn&& fn) {
  if constexpr (kernel_spec<KernelSpecs::Current>::ProfileEnabled) {
    if (fn(KernelSpecs::Current)) {
      return KernelSpecs::Current;
    };
  }
  if constexpr (!KernelSpecs::LastProfile) {
    return find_ct_profile_with_predicate<Fn, typename KernelSpecs::next>(std::forward<Fn>(fn));
  } else {
    return ct_profile::invalid;
  }
}

}  // namespace portfft::detail

#endif  // PORTFFT_COMMON_COMPILETIME_TUNING_PROFILE_HPP
