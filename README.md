# SYCL-FFT

## Introduction

SYCL-FFT is a library implementing Fast Fourier Transforms using SYCL and C++.
SYCL-FFT is in early stages of development and will support more options and optimizations in the future.

## Pre-requisites

* [DPC++] oneAPI release 2023.1.0
  * Nightly releases should work but are not tested
  * Other SYCL implementations are not tested
* [Level Zero] drivers
  * OpenCL drivers are not supported
* CMake 3.16+

## Getting Started

### Building with CMake

Clone SYCL-FFT and run the following commands from the cloned repository.

Build using DPC++ 2023.1.0 as:

```shell
source /opt/intel/oneapi/compiler/2023.1.0/env/vars.sh
cmake -Bbuild -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/clang++ -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/clang -DSYCLFFT_BUILD_TESTS=ON -DSYCLFFT_BUILD_BENCHMARKS=ON
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

SYCL-FFT currently requires to set the subgroup size at compile time. Multiple sizes can be set and the first one that is supported by the device will be used. Depending on the device used you may need to set the subgroup size with `-DSYCLFFT_SUBGROUP_SIZES=<comma separated list of sizes>`. By default only size 32 is used.
If you run into the exception with the message `None of the compiled subgroup sizes are supported by the device!` then `DSYCLFFT_SUBGROUP_SIZES` must be set to a different value(s) supported by the device.

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

## Supported configurations

SYCL-FFT is still in early development. The supported configurations are:

* complex to complex transforms
* single and double precisions
* forward and backward directions
* in-place and out-of-place transforms
* USM and buffer containers
* batched transforms
* 1D transforms only

The supported sizes depend on the CMake flags used which can be constrained by the device used.
`SYCLFFT_TARGET_REGS_PER_WI` is used to calculate the largest FFT that can fit in a workitem.
For instance setting it to `128` (resp. `256`) allows to fit a single precision FFT of size `27` (resp. `56`) in a single workitem.

The FFT sizes supported in the work-item, sub-group and work-group implementations are set using `SYCLFFT_COOLEY_TUKEY_OPTIMIZED_SIZES`.
The supported sizes are given as a comma-separated list of values.
By default, the size of $2^n$ and $2^n \times 3$ are enabled up to a value of 8192.

FFT sizes that are a product of a supported workitem FFT size and the subgroup size - the first value from `SYCLFFT_SUBGROUP_SIZES` that is supported by the device - are also supported.

Any batch size is supported as long as the input and output data fits in global memory.

By default the library assumes subgroup size of 32 is used. If that is not supported by the device it is running on, the subgroup size can be set using `SYCLFFT_SUBGROUP_SIZES`.

## Known issues

* Specialization constants are currently emulated on Nvidia and AMD backends. SYCL-FFT relies on this feature on Nvidia devices in particular so the performance is not optimal on these devices.

We are investigating other performance issues that affect all the backends.

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
[Level Zero]: https://dgpu-docs.intel.com/technologies/level-zero.html
[developer website]: https://developer.codeplay.com
[Codeplay Software Ltd]: https://www.codeplay.com
[DPC++ compiler documentation page]: https://intel.github.io/llvm-docs/UsersManual.html
