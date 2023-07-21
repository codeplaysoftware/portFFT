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
_header_path = Path("ref_data_include").joinpath("benchmark_reference.hpp")


def make_spec(file_base):
    return (make_header_spec(file_base), make_data_spec(file_base))


def make_header_spec(file_base):
    return {
        "file_path": str(Path(file_base).joinpath(_header_path)),
        "header_guard_name": "BENCHMARK_REFERENCE_GENERATED"
    }


def make_data_spec(file_base):

    def make_path(file_name):
        return str(Path(file_base).joinpath(_data_folder, file_name))

    cases = [
        # 1. small complex 1D fits in workitem Cooley-Tukey
        {
            "file_path": make_path("benchmark_1.dat"),
            "transform_type": "COMPLEX",
            "input_dimensions": [16],
            "batch": 8 * 1024 * 1024,
        },
        # 2. medium-small complex 1D fits in subgroup Cooley-Tukey
        {
            "file_path": make_path("benchmark_2.dat"),
            "transform_type": "COMPLEX",
            "input_dimensions": [256],
            "batch": 512 * 1024,
        },
        # 3. medium-large complex 1D fits in local memory Cooley-Tukey
        {
            "file_path": make_path("benchmark_3.dat"),
            "transform_type": "COMPLEX",
            "input_dimensions": [4 * 1024],
            "batch": 32 * 1024,
        },
        # 4. large complex 1D fits in global memory Cooley-Tukey
        {
            "file_path": make_path("benchmark_4.dat"),
            "transform_type": "COMPLEX",
            "input_dimensions": [64 * 1024],
            "batch": 2 * 1024,
        },
        # 5. large complex 1D fits in global memory Bluestein
        {
            "file_path": make_path("benchmark_5.dat"),
            "transform_type": "COMPLEX",
            "input_dimensions": [64 * 1024 + 1],
            "batch": 2 * 1024,
        },
        # 6. large complex 2D fits in global memory
        {
            "file_path": make_path("benchmark_6.dat"),
            "transform_type": "COMPLEX",
            "input_dimensions": [4096, 4096],
            "batch": 8,
        },
        # 7. small real 1D fits in workitem Cooley-Tukey
        {
            "file_path": make_path("benchmark_7.dat"),
            "transform_type": "REAL",
            "input_dimensions": [32],
            "batch": 8 * 1024 * 1024,
        },
        # 8. medium-small real 1D fits in subgroup Cooley-Tukey
        {
            "file_path": make_path("benchmark_8.dat"),
            "transform_type": "REAL",
            "input_dimensions": [512],
            "batch": 512 * 1024,
        },
        # 9. medium-large real 1D fits in local memory Cooley-Tukey
        {
            "file_path": make_path("benchmark_9.dat"),
            "transform_type": "REAL",
            "input_dimensions": [8 * 1024],
            "batch": 32 * 1024,
        },
        # 10. large real 1D fits in global memory Cooley-Tukey
        {
            "file_path": make_path("benchmark_10.dat"),
            "transform_type": "REAL",
            "input_dimensions": [128 * 1024],
            "batch": 2 * 1024,
        },
        # 11. small real 3D
        {
            "file_path": make_path("benchmark_11.dat"),
            "transform_type": "REAL",
            "input_dimensions": [64, 64, 64],
            "batch": 1024,
        }
    ]
    return cases
