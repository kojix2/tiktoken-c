#!/bin/sh
cc test.c -L ../target/debug/ -ltiktoken_c -o test
LD_LIBRARY_PATH="../target/debug" ./test "I am a cat."