#/***************************************************************************
# *
# *  @license
# *  Copyright (C) Codeplay Software Limited
# *  Licensed under the Apache License, Version 2.0 (the "License");
# *  you may not use this file except in compliance with the License.
# *  You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# *  For your convenience, a copy of the License has been included in this
# *  repository.
# *
# *  Unless required by applicable law or agreed to in writing, software
# *  distributed under the License is distributed on an "AS IS" BASIS,
# *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *  See the License for the specific language governing permissions and
# *  limitations under the License.
# *
# *  Codeplay's portFFT
# *
# *  @filename CMakeLists.txt
# *
# **************************************************************************/

include_guard()

include(FindPackageHandleStandardArgs)

function(portfft_manage_ct_profiles)
    set(ct_config_list "")
    if(${PORTFFT_ENABLED_KERNEL_CONFIGS} STREQUAL "auto")
        if("${CMAKE_CXX_FLAGS}" MATCHES "fsycl-targets=.*(nvptx64|nvidia_gpu)")
            list(APPEND ct_config_list "Nvidia")
        elseif("${CMAKE_CXX_FLAGS}" MATCHES "fsycl-targets=.*(amdgcn|amd_gpu)")
            list(APPEND ct_config_list "AMD")
        else()
            list(APPEND ct_config_list "Intel")
        endif()
    else()
        set(ct_config_list ${PORTFFT_ENABLED_KERNEL_CONFIGS})
    endif()

    if(ct_config_list STREQUAL "")
        message(FATAL_ERROR "PORTFFT_ENABLED_KERNEL_CONFIGS should not be empty")
    endif()

    foreach(ct_config ${ct_config_list})
        if(ct_config MATCHES "Intel.*")
            target_compile_definitions(portfft INTERFACE PORTFFT_BUILD_INTEL_KERNEL_PROFILE)
        elseif(ct_config MATCHES "AMD.*")
            target_compile_definitions(portfft INTERFACE PORTFFT_BUILD_AMD_KERNEL_PROFILE)
        elseif(ct_config MATCHES "Nvidia.*")
            target_compile_definitions(portfft INTERFACE PORTFFT_BUILD_NVIDIA_KERNEL_PROFILE)
        elseif(ct_config MATCHES "Custom.*")
            target_compile_definitions(portfft INTERFACE PORTFFT_BUILD_CUSTOM_KERNEL_PROFILE)
        else()
            message(FATAL_ERROR "Unknown compile-time configuration profile for portFFT: " ${ct_config})
        endif()
    endforeach()
endfunction()
