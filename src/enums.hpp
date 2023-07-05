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

#ifndef SYCL_FFT_ENUMS_HPP
#define SYCL_FFT_ENUMS_HPP

namespace sycl_fft {

enum class domain { REAL, COMPLEX };

enum class complex_storage { COMPLEX, REAL_REAL };

enum class placement { IN_PLACE, OUT_OF_PLACE };

enum class direction { FORWARD, BACKWARD };

namespace detail {
enum class pad { DO_PAD, DONT_PAD };

enum class level { WORKITEM, SUBGROUP, WORKGROUP, DEVICE };

enum class transpose { NOT_TRANSPOSED, TRANSPOSED };

enum class memory { BUFFER, USM };
}  // namespace detail

}  // namespace sycl_fft

#endif
