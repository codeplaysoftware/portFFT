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

namespace sycl_fft{

enum class domain {
   REAL,
   COMPLEX
};

enum class complex_storage {
    COMPLEX,
    REAL_REAL
};

enum class packed_format {
    COMPLEX,
    CONJUGATE_EVEN
};

enum class placement {
    IN_PLACE,
    OUT_OF_PLACE
};

}

#endif