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
# *  Codeplay's SYCL-FFT
# *
# *  @filename CMakeLists.txt
# *
# **************************************************************************/

# Clang-tidy needs to have a compilation data-base to work on, so we
# need a dummy project.
add_executable(clang_tidy_dummy clang_tidy_dummy.cpp)
add_sycl_to_target(TARGET clang_tidy_dummy)
target_link_libraries(
    clang_tidy_dummy
    PRIVATE
    portfft
    portfft_warnings
)
add_custom_target(clang_tidy_config DEPENDS ${CMAKE_SOURCE_DIR}/test/clang_tidy/.clang-tidy)
add_dependencies(clang_tidy_dummy clang_tidy_config)

# setup clang-tidy command from executable + options
get_filename_component(CXX_COMPILER_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
set(CLANG_TIDY_COMMAND "${CXX_COMPILER_DIR}/../bin-llvm/clang-tidy;--header-filter=${CMAKE_SOURCE_DIR}/src/*;--config-file=${CMAKE_SOURCE_DIR}/test/clang_tidy/.clang-tidy")
if(PORTFFT_CLANG_TIDY_AUTOFIX)
    list(APPEND CLANG_TIDY_COMMAND "--fix")
endif()

message(STATUS "Clang-tidy command: ${CLANG_TIDY_COMMAND}")

# set CXX_CLANG_TIDY property after defining the target
set_target_properties(clang_tidy_dummy PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")

