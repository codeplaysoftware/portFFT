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
 *  A tool to generate FFT reference data and headers detailing this data.
 *
 ************************************************************************"""
from generate_precomputed_fft_file import generate_and_write_data
from generate_verification_data_integration_header import generate_header_string
from verification_data_config import *
import benchmark_reference_specification
import test_reference_specification
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_path",
                        help="The path of the build directory for portFFT",
                        required=True)
    parser.add_argument("--verbose", help="Verbose", action='store_true')
    parser.add_argument("--data",
                        help="Generate the data.",
                        action='store_true')
    parser.add_argument("--header",
                        help="Generate the headers.",
                        action='store_true')
    parser.add_argument("--test",
                        help="Generate for unit tests",
                        action='store_true')
    parser.add_argument("--benchmark",
                        help="Generate for benchmarks",
                        action='store_true')
    parser.add_argument(
        "--just_list_outputs",
        help=
        "Prints the output files that this will produce. Don't generate them.",
        action='store_true')
    args = parser.parse_args()
    file_base = args.build_path
    be_verbose = args.verbose
    do_data = args.data
    do_header = args.header
    just_print_output_files = args.just_list_outputs

    specs = []
    if args.benchmark:
        specs.append(benchmark_reference_specification.make_spec(file_base))
    if args.test:
        specs.append(test_reference_specification.make_spec(file_base))

    if just_print_output_files:
        for spec in specs:
            if do_header:
                print(spec[0]["file_path"], end=';')
            if do_data:
                for config in spec[1]:
                    print(config["file_path"], end=';')
        exit()

    if do_data:
        for spec in specs:
            for config in spec[1]:
                generate_and_write_data(ValidationDataConfig(
                    config, validate_now=True),
                                        verbose=be_verbose)
    if do_header:
        for spec in specs:
            generate_header_string(spec[1], spec[0]["file_path"],
                                   spec[0]["header_guard_name"])
