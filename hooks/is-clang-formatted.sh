#!/bin/bash
#
# (C) Codeplay Software Ltd
#     SYCL-FFT source code formatting check script

set -euo pipefail
IFS=$'\n\t'

hooks/clang-format-all.sh
git diff --quiet
