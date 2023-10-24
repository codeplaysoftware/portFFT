# portFFT

## Introduction

portFFT is a library implementing Fast Fourier Transforms using SYCL and C++.
portFFT is in early stages of development and will support more options and optimizations in the future.

## Pre-requisites

* [DPC++] oneAPI release 2023.2.0
  * Nightly releases should work but are not tested
  * Other SYCL implementations are not tested
* [Level Zero] drivers
  * OpenCL drivers are not supported
* CMake 3.20+
* For tests and verifying benchmarks:
  * Python
  * Numpy

## Getting Started

### Building with CMake

Clone portFFT and run the following commands from the cloned repository.

Build using DPC++ 2023.2.0 as:

```shell
source /opt/intel/oneapi/compiler/2023.2.0/env/vars.sh
cmake -Bbuild -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2023.2.0/linux/bin/icpx -DPORTFFT_BUILD_TESTS=ON -DPORTFFT_BUILD_BENCHMARKS=ON
cmake --build build
```

Build using DPC++ nightlies as (SPIR64 target only):

```shell
cmake -Bbuild -DCMAKE_CXX_COMPILER=/path/to/dpcpp/bin/clang++ -DCMAKE_C_COMPILER=/path/to/dpcpp/bin/clang -DPORTFFT_BUILD_TESTS=ON -DPORTFFT_BUILD_BENCHMARKS=ON
cmake --build build
```

To compile AOT for a specific device, specify the target device with:

```shell
-DPORTFFT_DEVICE_TRIPLE=<T>[T1,..,Tn]
```

The list of available targets can be found on [DPC++ compiler documentation page].
Some AOT targets do not support double precision.
To disable the building of tests and benchmarks using double precision, set `-DPORTFFT_ENABLE_DOUBLE_BUILDS=OFF`.

portFFT currently requires to set the subgroup size at compile time. Multiple sizes can be set and the first one that is supported by the device will be used. Depending on the device used you may need to set the subgroup size with `-DPORTFFT_SUBGROUP_SIZES=<comma separated list of sizes>`. By default only size 32 is used.
If you run into the exception with the message `None of the compiled subgroup sizes are supported by the device!` then `DPORTFFT_SUBGROUP_SIZES` must be set to a different value(s) supported by the device.

### Tests

Tests are build if the CMake setting `PORTFFT_BUILD_TESTS` is set to `ON`.
Additionally, this enables `clang-tidy` checks if `PORTFFT_CLANG_TIDY` is at its default value of `ON`.
Automatic fixing of some `clang-tidy` warnings can be enabled by setting `PORTFFT_CLANG_TIDY_AUTOFIX` to `ON`.

Run the tests from the build folder with:

```shell
ctest
```

### portFFT benchmarks

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

portFFT is still in early development. The supported configurations are:

* complex to complex transforms
* single and double precisions
* forward and backward directions
* in-place and out-of-place transforms
* USM and buffer containers
* batched transforms
* 1D transforms only

The supported sizes depend on the CMake flags used which can be constrained by the device used.
`PORTFFT_VEC_LOAD_BYTES` is used to calculate the largest FFT that can fit in a workitem.
For instance setting it to `128` (resp. `256`) allows to fit a single precision FFT of size `27` (resp. `56`) in a single workitem.

FFT sizes that are a product of a supported workitem FFT size and the subgroup size - the first value from `PORTFFT_SUBGROUP_SIZES` that is supported by the device - are also supported.

Any batch size is supported as long as the input and output data fits in global memory.

By default the library assumes subgroup size of 32 is used. If that is not supported by the device it is running on, the subgroup size can be set using `PORTFFT_SUBGROUP_SIZES`.

## Known issues

* Specialization constants are currently emulated on Nvidia and AMD backends. portFFT relies on this feature on Nvidia devices in particular so the performance is not optimal on these devices.

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
