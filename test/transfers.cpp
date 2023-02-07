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
#include <common/transfers.hpp>
#include <complex>

constexpr int N = 8;
constexpr int sg_size = 32;
using ftype = double;

using complex_type = std::complex<ftype>;

class test_transfers_kernel;

void init(int size, complex_type* a){
    for(int i=0;i<size;i++){
        a[i] = {static_cast<ftype>(i), static_cast<ftype>(0/*(N-i)%11*/)};
    }
}

TEST(transfers, all){
    complex_type a[N*sg_size];
    complex_type b[N*sg_size];
    init(N*sg_size,a);

    sycl::queue q;
    complex_type* a_dev = sycl::malloc_device<complex_type>(N*sg_size,q);
    complex_type* b_dev = sycl::malloc_device<complex_type>(N*sg_size,q);
    complex_type* c_dev = sycl::malloc_device<complex_type>(N*sg_size,q);
    q.copy(a, a_dev, N*sg_size);
    q.copy(b, b_dev, N*sg_size);
    q.wait();

    q.submit([&](sycl::handler& h){
        sycl::local_accessor<complex_type,1> loc1(N*sg_size, h);
        sycl::local_accessor<complex_type,1> loc2(N*sg_size, h);
#ifdef SYCL_IMPLEMENTATION_ONEAPI
#define SUBGROUP_SIZE_ATTRIBUTE [[intel::reqd_sub_group_size(sg_size)]]
#else
#define SUBGROUP_SIZE_ATTRIBUTE
#endif
        h.parallel_for<test_transfers_kernel>(sycl::nd_range<1>({sg_size}, {sg_size}),[=](sycl::nd_item<1> it) SUBGROUP_SIZE_ATTRIBUTE {
            size_t local_id = it.get_sub_group().get_local_linear_id();

            complex_type priv[N];

            sycl_fft::global2local(a_dev, loc1, N*sg_size, sg_size, local_id);
            it.barrier();
            sycl_fft::local2private<N>(loc1, priv, local_id, N);
            sycl_fft::private2local<N>(priv, loc2, local_id, N);
            it.barrier();
            sycl_fft::local2global(loc2, b_dev, N*sg_size, sg_size, local_id);
        });
    });

    q.wait();

    q.copy(b_dev, b, N*sg_size);
    q.wait();

    for(int i=0;i<N*sg_size;i++){
        EXPECT_EQ(a[i].real(), b[i].real()) << "difference at index " << i;
        EXPECT_EQ(a[i].imag(), b[i].imag()) << "difference at index " << i;

    }
    std::cout << "done" <<std::endl;
}
