#include <stdio.h>
#include <stdlib.h>
#include "tiktoken.h"

int main()
{
  // Define the model
  const char *model = "gpt-4";

  // Create the messages
  CChatCompletionRequestMessage messages[3];

  messages[0].role = "system";
  messages[0].content = "You are a helpful assistant that only speaks French.";
  messages[0].name = NULL;
  messages[0].function_call = NULL;

  messages[1].role = "user";
  messages[1].content = "Hello, how are you?";
  messages[1].name = NULL;
  messages[1].function_call = NULL;

  messages[2].role = "system";
  messages[2].content = "Parlez-vous francais?";
  messages[2].name = NULL;
  messages[2].function_call = NULL;

  // Get the maximum tokens for chat completion
  size_t max_tokens = tiktoken_get_chat_completion_max_tokens(model, 3, messages);

  // Display the max_token parameter value
  printf("Max_token parameter value: %zu\n", max_tokens);

  return 0;
}
