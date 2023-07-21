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


class ValidationDataConfig:
    """A configuration for the validation FFT data file."""

    def __init__(self, config_dict=None, validate_now=False):
        # The constraints for a valid config are given in the valid_config()
        # function.
        self.file_path = config_dict.get("file_path", None)
        self.batch = config_dict.get("batch", None)
        self.transform_type = config_dict.get("transform_type", None)
        self.input_dimensions = config_dict.get("input_dimensions", None)
        self.max_input_size = config_dict.get("max_input_size",
                                              1024 * 1024 * 1024)
        if validate_now:
            self.throw_on_invalid_config()

    def throw_on_invalid_config(self):
        self.file_path = Path(self.file_path)
        # Confirm a sensible batch count.
        if not self.batch > 0:
            raise Exception("--batch argument must be +ve. Value was " +
                            str(self.batch))
        # Confirm a meaningful transform_type
        if not (self.transform_type == "REAL"
                or self.transform_type == "COMPLEX"):
            raise Exception("Invalid transform type set.")
        # Check validity of input dimensions.
        num_values = self.batch
        for dim in self.input_dimensions:
            if not dim > 0:
                raise Exception("Invalid dimension value: " + dim)
            num_values = num_values * dim
        if not num_values < self.max_input_size:
            raise Exception("batch * prod(dims) exceeds maximum value.")
