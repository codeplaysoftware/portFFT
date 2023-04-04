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

file(GLOB UNIT_TESTS ${PROJECT_SOURCE_DIR}/test/unit_test/*.cpp)

include(GoogleTest)
foreach(UNIT_TEST_FILE ${UNIT_TESTS})
    get_filename_component(FILE_NAME ${UNIT_TEST_FILE} NAME_WE)
    set(TEST_TARGET "test_${FILE_NAME}")
    add_executable(
        ${TEST_TARGET}
        ${UNIT_TEST_FILE}
    )
    add_sycl_to_target(TARGET ${TEST_TARGET} SOURCES ${UNIT_TEST_FILE})
    target_link_libraries(
        ${TEST_TARGET}
        PRIVATE
        sycl_fft
        GTest::gtest_main
        Threads::Threads
    )
    target_include_directories(${TEST_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/test/common)
    gtest_discover_tests(${TEST_TARGET} DISCOVERY_MODE PRE_TEST)
endforeach()
