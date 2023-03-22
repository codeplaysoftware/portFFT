# SYCL-FFT

## Introduction

SYCL-FFT is a library implementing Fast Fourier Transforms using SYCL and C++.
SYCL-FFT is in early stages of development and will support more options and optimizations in the future.

## Pre-requisites

* A SYCL implementation such as [DPC++].
* CMake

## Getting Started

Clone SYCL-FFT and run the following commands from the cloned repository.

Build using DPC++ 2023.0.0 as:

```shell
source /opt/intel/oneapi/compiler/2023.0.0/env/vars.sh
cmake -Bbuild -DCMAKE_CXX_COMPILER=${ONEAPI_ROOT}/compiler/2023.0.0/linux/bin-llvm/clang++ -DCMAKE_C_COMPILER=${ONEAPI_ROOT}/compiler/2023.0.0/linux/bin-llvm/clang -DSYCLFFT_BUILD_TESTS=ON -DSYCLFFT_BUILD_BENCHMARKS=ON
cmake --build build
```

Build using DPC++ nightlies as (SPIR64 target only):

```shell
cmake -Bbuild -DCMAKE_CXX_COMPILER=/path/to/dpcpp/bin/clang++ -DCMAKE_C_COMPILER=/path/to/dpcpp/bin/clang -DSYCLFFT_BUILD_TESTS=ON -DSYCLFFT_BUILD_BENCHMARKS=ON
cmake --build build
```

To compile AOT for a specific device, mention the target device as 

```
-DSYCLFFT_DEVICE_TRIPLE=<T>[T1,..,Tn]
```

The list of available targets can be found on [DPC++ compiler documentation page]

Run the tests from the build folder with:

```shell
ctest
```

Run the benchmarks from the build folder with:

```shell
./bench/bench_workitem
```

## Troubleshooting

The library should compile without error on our supported platforms.
If you run into trouble, or think you have found a bug, we have a support
forum available through the [ComputeCpp website], or create an issue on GitHub.

## Maintainers

This library is maintained by [Codeplay Software Ltd].
If you have any problems, please contact sycl@codeplay.com.

## Contributions

This library is licensed under the Apache 2.0 license. Patches are very
welcome! If you have an idea for a new feature or a fix, please get in
contact.

[DPC++]: https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top.html
[ComputeCpp website]: https://developer.codeplay.com
[Codeplay Software Ltd]: https://www.codeplay.com
[DPC++ compiler documentation page]: https://intel.github.io/llvm-docs/UsersManual.html
