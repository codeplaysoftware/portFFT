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
#include <common/workitem.hpp>
#include <descriptor.hpp>

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

#include <complex>

constexpr int N = 32;

using ftype = float;
using complex_type = std::complex<ftype>;

void init(complex_type* a, complex_type* b, complex_type* c){
    for(int i=0;i<N;i++){
        c[i] = b[i] = a[i] = {static_cast<ftype>(i), static_cast<ftype>((N-i)%11)};
    }
    std::cout << "init " << std::endl;
    for(int i=0;i<N;i++){
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
    for(int i=0;i<N;i++){
        max_err = std::max(max_err, error(a[i], b[i]));
        if(!eq(a[i], b[i])){
            err = true;
        }
    }
    std::cout << "max error: " << max_err << std::endl;
    if(err){
        for(int i=0;i<N;i++){
            std::cout << "(" <<a[i].real()<<","<<a[i].imag()<<"), ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
        for(int i=0;i<N;i++){
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

TEST(wi_dft, all){
    complex_type a[N];
    complex_type b[N];
    complex_type c[N];

    init(a,b,c);
    std::vector<std::complex<long double>> a_v(N);
    std::vector<std::complex<long double>> out_v(N);
    for(int i=0;i<N;i++){
        a_v[i] = {a[i].real(), a[i].imag()};
    }
    reference_forward_dft(a_v, out_v);
    std::cout << "ref " << std::endl;
    for(int i=0;i<N;i++){
        std::cout << out_v[i];
    }
    std::cout << std::endl;
    /*sycl::double2 a2[N];
    sycl::double2 b2[N];
    for(int i=0;i<4;i++){
        a2[i] = a[2*i];
        b2[i] = b[2*i];
    }
    cooley_tukey_dft<2,2,1,1>(a2,b2);
    cooley_tukey_dft<2,2,2,2>(a,b);
    naive_dft<4,2,2>(a,c);*/

    sycl_fft::wi_dft<N,1,1>(reinterpret_cast<ftype*>(b),reinterpret_cast<ftype*>(b));

    sycl_fft::detail::naive_dft<N,1,1>(reinterpret_cast<ftype*>(a),reinterpret_cast<ftype*>(c));

    std::cout << std::endl;
    std::cout << "comparison with naive" << std::endl;
    check(b,c);
    std::cout << "comparison with reference" << std::endl;
    std::vector<complex_type> out_ref(N);
    for(int i=0;i<N;i++){
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

TEST(workitem_impl, descriptor){
    constexpr int N_transforms = 133;
    complex_type a[N*N_transforms];
    complex_type b[N*N_transforms];
    complex_type c[N*N_transforms];
    init(a,b,c);

    sycl::queue q;
    complex_type* a_dev = sycl::malloc_device<complex_type>(N*N_transforms,q);
    complex_type* b_dev = sycl::malloc_device<complex_type>(N*N_transforms,q);
    q.copy(a, a_dev, N*N_transforms);
    q.copy(b, b_dev, N*N_transforms);
    q.wait();

    sycl_fft::descriptor<ftype, sycl_fft::domain::COMPLEX> desc{{N}};
    auto committed = desc.commit(q);
    committed.compute_forward(b_dev);

    q.wait();
    q.copy(b_dev, b, N*N_transforms);
    q.wait();
    
    std::cout << "comparison with reference" << std::endl;
    for(int j=0;j<N_transforms;j++){
        std::vector<std::complex<long double>> a_v(N);
        std::vector<std::complex<long double>> out_v(N);
        for(int i=0;i<N;i++){
            a_v[i] = {a[i + j*N].real(), a[i + j*N].imag()};
        }
        reference_forward_dft(a_v, out_v);
        std::vector<complex_type> out_ref(N);
        for(int i=0;i<N;i++){
            out_ref[i] = {static_cast<ftype>(out_v[i].real()), static_cast<ftype>(out_v[i].imag())};
        }
        EXPECT_TRUE(check(b + j*N,out_ref.data())) << "transform " << j;
    }

}
