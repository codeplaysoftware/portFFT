
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
 *  Codeplay's SYCL-FFT
 *
 **************************************************************************/
namespace sycl_fft::detail{

/**
 * Determines the largest factor of N that is smaller than or equal to sqrt(N) - in static constexpr member factor.
 * @tparam N size to factorize
*/
template<int N>
struct factorize{
    static_assert(N!=N, "not implemented");
};

#define IMPL(NUM, FACTOR) \
template<> \
struct factorize<NUM>{ \
    static constexpr int factor = FACTOR; \
};

IMPL(1,1)
IMPL(2,1)
IMPL(3,1)
IMPL(4,2)
IMPL(5,1)
IMPL(6,2)
IMPL(7,1)
IMPL(8,2)
IMPL(9,3)
IMPL(10,2)
IMPL(11,1)
IMPL(12,3)
IMPL(13,1)
IMPL(14,2)
IMPL(15,3)
IMPL(16,4)
IMPL(17,1)
IMPL(18,3)
IMPL(19,1)
IMPL(20,4)
IMPL(21,3)
IMPL(22,2)
IMPL(23,1)
IMPL(24,4)
IMPL(25,5)
IMPL(26,2)
IMPL(27,3)
IMPL(28,4)
IMPL(29,1)
IMPL(30,5)
IMPL(31,1)
IMPL(32,4)
IMPL(33,3)
IMPL(34,2)
IMPL(35,5)
IMPL(36,6)
IMPL(37,1)
IMPL(38,2)
IMPL(39,3)
IMPL(40,5)
IMPL(41,1)
IMPL(42,6)
IMPL(43,1)
IMPL(44,4)
IMPL(45,5)
IMPL(46,2)
IMPL(47,1)
IMPL(48,6)
IMPL(49,7)
IMPL(50,5)
IMPL(51,3)
IMPL(52,4)
IMPL(53,1)
IMPL(54,6)
IMPL(55,5)
IMPL(56,7)
IMPL(57,3)
IMPL(58,2)
IMPL(59,1)
IMPL(60,6)
IMPL(61,1)
IMPL(62,2)
IMPL(63,7)
IMPL(64,8)
// 64 is likely the largest size we will be able to handle within one workitem on current GPUs

};

#undef IMPL