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
#include <gtest/gtest.h>
#include <common/factorize.hpp>

int factorize(int N){
    int res = 1;
    for(int i=2;i*i<=N;i++){
        if(N%i==0){
            res=i;
        }
    }
    return res;
}

template<int N>
void test(){
    int factor = sycl_fft::detail::factorize<N>::factor;
    int correct = factorize(N);
    EXPECT_EQ(factor, correct) << "error N: " << N << std::endl;
    if constexpr(N - 1 > 0){ test<N-1>(); }
}

TEST(factorize, all){
    test<64>();
}
