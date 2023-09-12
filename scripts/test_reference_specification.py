"""************************************************************************
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
 *  Codeplay's portFFT
 *
 ************************************************************************"""
from pathlib import Path

# This file defines the specification for the benchmark cases.
# It must match the specification given in test/bench/utils/reference_dft_set.hpp.

_data_folder = Path("ref_data")
_header_path = Path("ref_data_include").joinpath("test_reference.hpp")


def _gen_configuration_1d(file_base, description, domain, batch_sizes,
                          fft_sizes):
    """Generate configurations where a single file can be used for multiple
    batch sizes. 
    """

    # Only use the max batch size.
    max_batch = max(batch_sizes)
    # Generate a config per FFT size.
    i = 0
    configs = []
    for fft_size in fft_sizes:
        configs.append({
            "file_path":
            Path(file_base).joinpath(_data_folder,
                                     "test_" + description + str(i) + ".dat"),
            "transform_type":
            domain,
            "input_dimensions": [fft_size],
            "batch":
            max_batch
        })
        i += 1
    return configs


def make_spec(file_base):
    return (make_header_spec(file_base), make_data_spec(file_base))


def make_header_spec(file_base):
    return {
        "file_path": str(Path(file_base).joinpath(_header_path)),
        "header_guard_name": "TEST_REFERENCE_GENERATED"
    }


def make_data_spec(file_base):
    configs = []
    configs += _gen_configuration_1d(file_base, "workItemTest", "COMPLEX",
                                     [1, 3, 33000], [1, 2, 3, 4, 8, 9])
    configs += _gen_configuration_1d(file_base, "workItemOrSubgroupTest",
                                     "COMPLEX", [1, 3, 555], [16, 32])
    configs += _gen_configuration_1d(file_base, "SubgroupTest", "COMPLEX",
                                     [1, 3, 555], [64, 96, 128])
    configs += _gen_configuration_1d(file_base, "SubgroupOrWorkgroupTest",
                                     "COMPLEX", [3], [256, 512, 1024])
    configs += _gen_configuration_1d(file_base, "WorkgroupTest", "COMPLEX",
                                     [1, 3], [2048, 3072, 4096])
    configs += _gen_configuration_1d(file_base, "GlobalTest", "COMPLEX",
                                     [1, 3], [16384, 32768, 65536])
    return configs
