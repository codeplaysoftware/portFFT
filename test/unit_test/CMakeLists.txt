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

find_package(SYCL)

# Get google test.
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(PORTFFT_UNIT_TESTS
    fft_float.cpp
    descriptor.cpp
)
if(PORTFFT_ENABLE_DOUBLE_BUILDS)
    list(APPEND PORTFFT_UNIT_TESTS
        fft_double.cpp
    )
endif()

include(GoogleTest)
foreach(UNIT_TEST_FILE ${PORTFFT_UNIT_TESTS})
    get_filename_component(FILE_NAME ${UNIT_TEST_FILE} NAME_WE)
    set(TEST_TARGET "test_${FILE_NAME}")
    add_executable(
        ${TEST_TARGET}
        ${UNIT_TEST_FILE}
    )
    add_sycl_to_target(TARGET ${TEST_TARGET})
    target_link_libraries(
        ${TEST_TARGET}
        PRIVATE
        portfft
        portfft_warnings
        GTest::gtest_main
        Threads::Threads
    )
    target_include_directories(${TEST_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/test/common)
    gtest_discover_tests(${TEST_TARGET} XML_OUTPUT_DIR output DISCOVERY_MODE PRE_TEST)
endforeach()
