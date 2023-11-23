#!/bin/sh

# Add missing import
export LD_LIBRARY_PATH="../target/debug"

cc test.c -L ../target/debug/ -ltiktoken_c -o test

OUTPUT=$(echo "I am a cat." | ./test) 

EXPECTED="40 1097 264 8415 13"

echo "Expected: $EXPECTED"
echo "     Got: $OUTPUT"

if [ "$OUTPUT" = "$EXPECTED" ]; then
  echo "Test passed successfully"
else
  echo "Test failed :("
  exit 1
fi
