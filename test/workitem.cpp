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
#include <common/workitem.hpp>
#include <complex>

constexpr int N = 32;
constexpr int M = 1;

using ftype = float;
using complex_type = std::complex<ftype>;

void init(complex_type* a, complex_type* b, complex_type* c){
    for(int i=0;i<N*M;i++){
        c[i] = b[i] = a[i] = {static_cast<ftype>(i), static_cast<ftype>((N-i)%11)};
    }
    std::cout << "init " << std::endl;
    for(int i=0;i<N*M;i++){
        std::cout << "(" <<a[i].real()<<","<<a[i].imag()<<"), ";
    }
    std::cout << std::endl;
}

ftype error(complex_type a, complex_type b){
    return std::abs(a.real()-b.real()) + std::abs(a.imag()-b.imag());
}

bool eq(complex_type a, complex_type b){
    return error(a,b) < 0.02;
}

bool check(complex_type* a, complex_type* b){
    bool err = false;
    ftype max_err = 0;
    for(int i=0;i<N*M;i++){
        max_err = std::max(max_err, error(a[i], b[i]));
        if(!eq(a[i], b[i])){
            err = true;
        }
    }
    std::cout << "max error: " << max_err << std::endl;
    if(err){
        for(int i=0;i<N*M;i++){
            std::cout << "(" <<a[i].real()<<","<<a[i].imag()<<"), ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
        for(int i=0;i<N*M;i++){
            std::cout << "(" <<b[i].real()<<","<<b[i].imag()<<"), ";
        }
        std::cout << std::endl;
    }
    return !err;
}

template <typename TypeIn, typename TypeOut>
void reference_forward_dft(std::vector<TypeIn> &in, std::vector<TypeOut> &out) {
    long double TWOPI = 2.0l * std::atan(1.0l) * 4.0l;

    std::complex<long double> out_temp; // Do the calculations using long double
    size_t N = out.size();
    for (int k = 0; k < N; k++) {
        out_temp = 0;
        for (int n = 0; n < N; n++) {
            auto multiplier = std::complex<long double>{ std::cos(n * k * TWOPI / N),
                                                  -std::sin(n * k * TWOPI / N) };
            out_temp += static_cast<std::complex<long double>>(in[n]) *
                        multiplier;
        }
        out[k] = static_cast<TypeOut>(out_temp);
    }
}

TEST(workitem, all){
    complex_type a[N*M];
    complex_type b[N*M];
    complex_type c[N*M];

    init(a,b,c);
    std::vector<std::complex<long double>> a_v(N*M);
    std::vector<std::complex<long double>> out_v(N*M);
    for(int i=0;i<N*M;i++){
        a_v[i] = {a[i].real(), a[i].imag()};
    }
    reference_forward_dft(a_v, out_v);
    std::cout << "ref " << std::endl;
    for(int i=0;i<N*M;i++){
        std::cout << out_v[i];
    }
    std::cout << std::endl;
    /*sycl::double2 a2[N*M];
    sycl::double2 b2[N*M];
    for(int i=0;i<4;i++){
        a2[i] = a[2*i];
        b2[i] = b[2*i];
    }
    cooley_tukey_dft<2,2,1,1>(a2,b2);
    cooley_tukey_dft<2,2,2,2>(a,b);
    naive_dft<4,2,2>(a,c);*/

    sycl_fft::wi_dft<N*M,1,1>(reinterpret_cast<ftype*>(b),reinterpret_cast<ftype*>(b));

    sycl_fft::detail::naive_dft<N*M,1,1>(reinterpret_cast<ftype*>(a),reinterpret_cast<ftype*>(c));

    std::cout << std::endl;
    std::cout << "comparison with naive" << std::endl;
    check(b,c);
    std::cout << "comparison with reference" << std::endl;
    std::vector<complex_type> out_ref(N*M);
    for(int i=0;i<N*M;i++){
        out_ref[i] = {static_cast<ftype>(out_v[i].real()), static_cast<ftype>(out_v[i].imag())};
    }
    bool res  = check(b,out_ref.data());
    EXPECT_TRUE(res);
/*
    init(a,b);
    cooley_tukey_dft<8,8,1>(a);
    naive_dft<64,1>(b);
    std::cout << check(a,b);

    init(a,b);
    cooley_tukey_dft<3,17,1>(a);
    naive_dft<51,1>(b);
    std::cout << check(a,b);*/
}
