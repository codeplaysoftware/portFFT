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
import numpy as np
from verification_data_config import *
from sys import stdout

def generate_data(validation_config, verbose=False):
    """For a validation config, generate random input data and compute the 
    FFT. returns a tuple of (inputData, outputData). Verbose prints information on what
    the function is doing.
    """

    dims = validation_config.input_dimensions
    batch = validation_config.batch
    dataGenDims = [batch] + dims
    if verbose:
        print("Generating data for " + str(dims) + " x " + str(batch) +
              " in " + validation_config.transform_type + " domain.")
    fileInData = np.random.uniform(-1, 1, dataGenDims).astype(np.double)
    if validation_config.transform_type == "COMPLEX":
        fileInData = fileInData + 1j * np.random.uniform(
            -1, 1, dataGenDims).astype(np.double)
    fileOutData = np.fft.fftn(fileInData, axes=range(1, len(dims) + 1))
    fileInData.reshape(-1, 1)
    fileOutData.reshape(-1, 1)
    return fileInData, fileOutData


def generate_and_write_data(validation_config, verbose=False):
    """For a validation config, generate random input data, compute the 
    FFT and write both to file. Verbose prints information on what
    the function is doing.
    """

    fileInData, fileOutData = generate_data(validation_config, verbose)
    # Now output to file - write binary files since the files are typically
    # multiple GB in size.
    if verbose:
        print("\tWriting to file " + str(validation_config.file_path))
    try:
        validation_config.file_path.parent.mkdir(exist_ok=True, parents=True)
        validation_config.file_path.touch()
    except Exception as e:
        raise Exception("Could not write header file to " +
                        str(validation_config.file_path))

    with validation_config.file_path.open(mode="wb") as f:
        fileInData.tofile(f)
        fileOutData.tofile(f)

def generate_and_dump_data(batch, input_dimensions, transform_type):
    a, b = generate_data(ValidationDataConfig({'batch':batch, 'input_dimensions':input_dimensions, 'transform_type':transform_type}))
    stdout.buffer.write(a.tobytes())
    stdout.buffer.write(b.tobytes())
