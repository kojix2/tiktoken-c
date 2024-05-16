#!/bin/sh

# Add missing import
export LD_LIBRARY_PATH="../target/debug"

cc test.c -L ../target/debug/ -ltiktoken_c -o test

# Initialize test result
ALL_TESTS_PASSED=true

# Test with gpt-4 model
OUTPUT_GPT4=$(echo "I am a cat." | ./test -m "gpt-4")
EXPECTED_GPT4="40 1097 264 8415 13"

echo "Testing gpt-4 model"
echo "Expected: $EXPECTED_GPT4"
echo "     Got: $OUTPUT_GPT4"

if [ "$OUTPUT_GPT4" = "$EXPECTED_GPT4" ]; then
  echo "Test passed successfully for gpt-4"
else
  echo "Test failed for gpt-4 :("
  ALL_TESTS_PASSED=false
fi

# Test with gpt-4o model
OUTPUT_GPT4O=$(echo "I am a cat." | ./test -m "gpt-4o")
EXPECTED_GPT4O="40 939 261 9059 13"

echo "Testing gpt-4o model"
echo "Expected: $EXPECTED_GPT4O"
echo "     Got: $OUTPUT_GPT4O"

if [ "$OUTPUT_GPT4O" = "$EXPECTED_GPT4O" ]; then
  echo "Test passed successfully for gpt-4o"
else
  echo "Test failed for gpt-4o :("
  ALL_TESTS_PASSED=false
fi

# Final test result
if [ "$ALL_TESTS_PASSED" = true ]; then
  echo "All tests passed successfully"
  exit 0
else
  echo "Some tests failed :("
  exit 1
fi
