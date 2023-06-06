#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

constexpr int N = 1024*1024*256;
constexpr int sg_size = 16;

template<int Shuffle>
void transfer(benchmark::State& state){
    sycl::queue q;
    float* a = sycl::malloc_device<float>(N, q);
    float* b = sycl::malloc_device<float>(N, q);

    for (auto _ : state) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>({N}, {sg_size}), [=](sycl::nd_item<1> it) 
                    [[sycl::reqd_sub_group_size(sg_size)]] {
                unsigned int gid = it.get_global_id(0);
                sycl::sub_group sg = it.get_sub_group();
                unsigned int lid = sg.get_local_id();
                float p = a[gid];
                #pragma(unroll)
                for(int i=0;i<Shuffle;i++){
                    //p += sycl::select_from_group(sg, p, lid + 1);
                    p += sycl::permute_group_by_xor(sg, p, 1);
                }
                b[gid] = p;
            });
        }).wait();
    }
}

BENCHMARK(transfer<0>);
BENCHMARK(transfer<7>);
BENCHMARK(transfer<14>);
BENCHMARK(transfer<100>);

BENCHMARK_MAIN();