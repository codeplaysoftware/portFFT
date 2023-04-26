# SYCL-FFT

## Introduction

SYCL-FFT is a library implementing Fast Fourier Transforms using SYCL and C++.
SYCL-FFT is in early stages of development and will support more options and optimizations in the future.

## Pre-requisites

* A SYCL implementation such as [DPC++].
* CMake

## Getting Started

### Building with CMake

Clone SYCL-FFT and run the following commands from the cloned repository.

Build using DPC++ 2023.1.0 as:

```shell
source /opt/intel/oneapi/compiler/2023.1.0/env/vars.sh
cmake -Bbuild -DCMAKE_CXX_COMPILER=${ONEAPI_ROOT}/compiler/2023.1.0/linux/bin-llvm/clang++ -DCMAKE_C_COMPILER=${ONEAPI_ROOT}/compiler/2023.1.0/linux/bin-llvm/clang -DSYCLFFT_BUILD_TESTS=ON -DSYCLFFT_BUILD_BENCHMARKS=ON
cmake --build build
```

Build using DPC++ nightlies as (SPIR64 target only):

```shell
cmake -Bbuild -DCMAKE_CXX_COMPILER=/path/to/dpcpp/bin/clang++ -DCMAKE_C_COMPILER=/path/to/dpcpp/bin/clang -DSYCLFFT_BUILD_TESTS=ON -DSYCLFFT_BUILD_BENCHMARKS=ON
cmake --build build
```

To compile AOT for a specific device, specify the target device with:

```shell
-DSYCLFFT_DEVICE_TRIPLE=<T>[T1,..,Tn]
```

The list of available targets can be found on [DPC++ compiler documentation page].
Some AOT targets do not support double precision.
To disable the building of tests and benchmarks using double precision, set `-DSYCLFFT_ENABLE_DOUBLE_BUILDS=OFF`.

SYCL-FFT currently requires to set the subgroup size at compile time. Depending on the device used you may need to set the subgroup size with `-DSYCLFFT_TARGET_SUBGROUP_SIZE=<size>`.
If you run into the exception with the message `Subgroup size <N1> of the [..] kernel does not match required size of <N2>` then `SYCLFFT_TARGET_SUBGROUP_SIZE` must be set to `N1`.

### Tests

Run the tests from the build folder with:

```shell
ctest
```

### SYCL-FFT benchmarks

Run pre-defined benchmarks from the build folder with:

```shell
./test/bench/bench_float
```

Run manual benchmarks from the build folder with for instance:

```shell
./test/bench/bench_manual_float d=cpx,n=5
```

Use the `--help` flag to print help message on the configuration syntax.

## Third party benchmarks

### [Open-source oneMKL]

The benchmark can be enabled with `-DSYCLFFT_INTEL_OPEN_ONEMKL_BENCHMARK_BACKEND=<backend>` where `<backend>` can be one of `MKLCPU`, `MKLGPU` or `CUFFT`.

The compiler must be set to `icpx` and `icx` using `-DCMAKE_CXX_COMPILER=${ONEAPI_ROOT}/compiler/2023.1.0/linux/bin/icpx -DCMAKE_C_COMPILER=${ONEAPI_ROOT}/compiler/2023.1.0/linux/bin/icx`.

Run the benchmark with:

```shell
./test/bench/bench_open_onemkl
```

### [Closed-source oneMKL]

The benchmark can be enabled with `-DSYCLFFT_ENABLE_INTEL_CLOSED_ONEMKL_BENCHMARKS=ON`.

The compiler must be set to `icpx` and `icx` using `-DCMAKE_CXX_COMPILER=${ONEAPI_ROOT}/compiler/2023.1.0/linux/bin/icpx -DCMAKE_C_COMPILER=${ONEAPI_ROOT}/compiler/2023.1.0/linux/bin/icx`.

Run the benchmark with:

```shell
./test/bench/bench_closed_onemkl
```

### [cuFFT]

The benchmark can be enabled with `-DSYCLFFT_ENABLE_CUFFT_BENCHMARKS=ON`.

Run the benchmark with:

```shell
./test/bench/bench_cufft
```

### [rocFFT]

The benchmark can be enabled with `-DSYCLFFT_ENABLE_ROCFFT_BENCHMARKS=ON`. ROCm 5.4.3 or greater is required.

The compiler must be set to `icpx` and `icx` using `-DCMAKE_CXX_COMPILER=${ONEAPI_ROOT}/compiler/2023.1.0/linux/bin/icpx -DCMAKE_C_COMPILER=${ONEAPI_ROOT}/compiler/2023.1.0/linux/bin/icx`.

Run the benchmark with:

```shell
./test/bench/bench_rocfft
```

## Troubleshooting

The library should compile without error on our supported platforms.
If you run into trouble, or think you have found a bug, we have a support
forum available through the [developer website], or create an issue on GitHub.

## Maintainers

This library is maintained by [Codeplay Software Ltd].
If you have any problems, please contact sycl@codeplay.com.

## Contributions

This library is licensed under the Apache 2.0 license. Patches are very
welcome! If you have an idea for a new feature or a fix, please get in
contact.

[DPC++]: https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top.html
[developer website]: https://developer.codeplay.com
[Codeplay Software Ltd]: https://www.codeplay.com
[DPC++ compiler documentation page]: https://intel.github.io/llvm-docs/UsersManual.html
[open-source oneMKL]: https://github.com/oneapi-src/oneMKL
[closed-source oneMKL]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html
[cuFFT]: https://docs.nvidia.com/cuda/cufft/
[rocFFT]: https://github.com/ROCmSoftwarePlatform/rocFFT
