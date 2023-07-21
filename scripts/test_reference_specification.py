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

# Ensure all filenames are unique
_file_counter = 0

def _append_configuration_1d(configs, file_base, description, domain, batch_sizes,
                             fft_sizes):
    """Generate configurations where a single file can be used for multiple
    batch sizes.
    """
    global _file_counter

    # Only use the max batch size.
    max_batch = max(batch_sizes)
    # Generate a config per FFT size.
    for fft_size in fft_sizes:
        filename = Path(file_base).joinpath(_data_folder, "test_" + description + str(_file_counter) + ".dat")
        _file_counter += 1
        # Skip if a configuration with the same expected result is already added
        # Find a configuration with the same domain and size
        added_config = next((c for c in configs if domain == c["transform_type"] and [fft_size] == c["input_dimensions"]), None)
        if added_config:
            # Updated filename and max_batch in existing config if needed
            if added_config["batch"] < max_batch:
                added_config["file_path"] = filename
                added_config["batch"] = max_batch
            continue
        # Add new config
        config = {}
        config["file_path"] = filename
        config["transform_type"] = domain
        config["input_dimensions"] = [fft_size]
        config["batch"] = max_batch
        configs.append(config)
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
    # Configurations must match with the test suites in test/unit_test/instantiate_fft_tests.hpp.
    # Only configurations with unique expected results are appended.
    _append_configuration_1d(configs, file_base, "workItemTest", "COMPLEX",
                             [1, 3, 33000], [1, 2, 3, 4, 8, 9])
    _append_configuration_1d(configs, file_base, "workItemOrSubgroupTest",
                             "COMPLEX", [1, 3, 555], [16, 32])
    _append_configuration_1d(configs, file_base, "SubgroupTest", "COMPLEX",
                             [1, 3, 555], [64, 96, 128])
    _append_configuration_1d(configs, file_base, "SubgroupOrWorkgroupTest",
                             "COMPLEX", [3], [256, 512, 1024])
    _append_configuration_1d(configs, file_base, "WorkgroupTest", "COMPLEX",
                             [1, 3], [2048, 3072, 4096])
    # The expected results are scaled during the tests for backward FFTs
    # so we do not need to specify the FFTs are backward.
    _append_configuration_1d(configs, file_base, "BackwardTest", "COMPLEX",
                             [1], [8, 9, 16, 32, 64, 4096])
    # The inputs and outputs are re-ordered during the tests
    # so we do not need to specify the stride or distance parameters.
    _append_configuration_1d(configs, file_base, "StridedTest", "COMPLEX",
                             [1, 3], [4, 128, 4096])
    _append_configuration_1d(configs, file_base, "Distance*Test", "COMPLEX",
                             [2, 50], [4, 128, 4096])
    _append_configuration_1d(configs, file_base, "StridedDistanceWorkItem*Test", "COMPLEX",
                             [5], [16])
    _append_configuration_1d(configs, file_base, "StridedDistanceSubgroup*Test", "COMPLEX",
                             [3], [128])
    _append_configuration_1d(configs, file_base, "StridedDistanceWorkgroup*Test", "COMPLEX",
                             [2], [4096])
    _append_configuration_1d(configs, file_base, "BatchInterleavedWorkItemTest", "COMPLEX",
                             [3], [16])
    _append_configuration_1d(configs, file_base, "BatchInterleavedSubgroupTest", "COMPLEX",
                             [3], [128])
    _append_configuration_1d(configs, file_base, "BatchInterleavedWorkgroupTest", "COMPLEX",
                             [3], [4096])
    _append_configuration_1d(configs, file_base, "BatchInterleavedLargerStrideDistanceTest", "COMPLEX",
                             [20], [16])
    _append_configuration_1d(configs, file_base, "ArbitraryInterleavedTest", "COMPLEX",
                             [4], [4])
    _append_configuration_1d(configs, file_base, "OverlapReadFwdTest", "COMPLEX",
                             [3], [4])
    _append_configuration_1d(configs, file_base, "OverlapReadBwdTest", "COMPLEX",
                             [3], [4])
    return configs
