#!/bin/bash
#
# (C) Codeplay Software Ltd
#     portFFT source code formatting script
#
#     NB: portFFT uses clang-format 11.
#

set -euo pipefail
IFS=$'\n\t'

git ls-files | grep -E "*\.h$|*\.hpp$|*\.cc$|*\.cpp$" | \
  xargs --max-procs=`nproc` --max-args=1 clang-format-11 -style=file -i
