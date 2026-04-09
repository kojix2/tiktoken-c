#!/bin/sh

set -eu

BUILD_DIR="${BUILD_DIR:-debug}"
CC="${CC:-cc}"

export LD_LIBRARY_PATH="../target/${BUILD_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="../target/${BUILD_DIR}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

"$CC" run_tests.c -L "../target/${BUILD_DIR}" -ltiktoken_c -o run_tests
./run_tests
