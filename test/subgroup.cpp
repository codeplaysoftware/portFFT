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
#include <common/subgroup.hpp>
#include <common/workitem.hpp>
#include <common/transfers.hpp>
#include <iostream>
#include <complex>

constexpr int N = 6;
constexpr int sg_size = 16;
//constexpr int stride = sg_size / N;

using ftype = float;
using complex_type = std::complex<ftype>;

void init(complex_type* a, complex_type* b, complex_type* c){
    for(int i=0;i<sg_size;i++){
        c[i] = b[i] = a[i] = {static_cast<ftype>(i), static_cast<ftype>((N-i)%11)};
    }
    std::cout << "init " << std::endl;
    for(int i=0;i<sg_size;i++){
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
    for(int i=0;i<sg_size;i++){
        max_err = std::max(max_err, error(a[i], b[i]));
        if(!eq(a[i], b[i])){
            err = true;
        }
    }
    std::cout << "max error: " << max_err << std::endl;
    if(err){
        for(int i=0;i<sg_size;i++){
            std::cout << "(" <<a[i].real()<<","<<a[i].imag()<<"), ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
        for(int i=0;i<sg_size;i++){
            std::cout << "(" <<b[i].real()<<","<<b[i].imag()<<"), ";
        }
        std::cout << std::endl;
    }
    return !err;
}

int main(){
    complex_type a[sg_size];
    complex_type b[sg_size];
    complex_type c[sg_size];
    init(a,b,c);
    c[2]=-999;

    sycl::queue q;
    complex_type* a_dev = sycl::malloc_device<complex_type>(sg_size,q);
    complex_type* b_dev = sycl::malloc_device<complex_type>(sg_size,q);
    q.copy(a, a_dev, sg_size);
    //q.copy(c, c_dev, sg_size);

    std::vector<std::complex<long double>> a_v(sg_size);
    std::vector<std::complex<long double>> out_v(sg_size);
    for(int i=0;i<sg_size;i++){
        a_v[i] = {a[i].real(), a[i].imag()};
    }

    q.wait();
    std::cout << "before kernel" << std::endl;
    q.submit([&](sycl::handler& h){
            sycl::local_accessor<complex_type,1> loc(sg_size, h);
            h.parallel_for(sycl::nd_range<1>({sg_size}, {sg_size}),
                    [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(sg_size)]] {
                sycl::sub_group sg = it.get_sub_group();
                size_t local_id = sg.get_local_linear_id();

                sycl_fft::global2local(a_dev, loc, sg_size, sg_size, local_id);
                it.barrier();
                sycl_fft::detail::cross_sg_dft<N,1>(reinterpret_cast<ftype*>(loc.get_pointer().get() + local_id)[0],
                                                    reinterpret_cast<ftype*>(loc.get_pointer().get() + local_id)[1],
                                                    sg);
                //loc[0] = 0;
                //loc[1] = 0;
                it.barrier();
                sycl_fft::local2global(loc, b_dev, sg_size, sg_size, local_id);
            });
        }).wait_and_throw();
    std::cout << "after kernel" << std::endl;
    q.copy(b_dev, b, sg_size).wait_and_throw();

    for(int i=0;i<sg_size;i+=N){
        //sycl_fft::detail::naive_dft<N,1,1>(reinterpret_cast<ftype*>(a + i),reinterpret_cast<ftype*>(c + i));
        sycl_fft::wi_dft<N,1,1>(reinterpret_cast<ftype*>(a + i),reinterpret_cast<ftype*>(c + i));
    }

    std::cout << std::endl;
    std::cout << "comparison with workitem" << std::endl;
    bool res  = check(b,c);
    std::cout << "is correct: " << res << std::endl;
}