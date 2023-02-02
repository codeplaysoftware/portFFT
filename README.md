Build using DPC++ 2023.0.0 as:
```
source /opt/intel/oneapi/compiler/2023.0.0/env/vars.sh
cmake -GNinja path/to/sycl-fft/ -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2023.0.0/linux/bin/icpx -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2023.0.0/linux/bin/icx -DSYCLFFT_BUILD_TESTS=ON -DSYCLFFT_BUILD_BENCHMARKS=ON
ninja
```

Build using DPC++ nightlies as (SPIR64 target only):
```
cmake -GNinja path/to/sycl-fft/ -DCMAKE_CXX_COMPILER=/home/hugh/dev/dpcpprel/20221020/bin/clang++ -DCMAKE_C_COMPILER=/home/hugh/dev/dpcpprel/20221020/bin/clang -DSYCLFFT_BUILD_TESTS=ON -DSYCLFFT_BUILD_BENCHMARKS=ON
ninja
```

Build with ComputeCpp as (currently non-working due to illegal SPIR causing ICE):
```
cmake -GNinja ../sycl-fft/ -DComputeCpp_DIR=/home/hugh/dev/computecpprel/20220912 -DSYCLFFT_BUILD_TESTS=ON -DSYCLFFT_BUILD_BENCHMARKS=ON
ninja
```

