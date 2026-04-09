#include <stdio.h>
#include <stdlib.h>
#include "tiktoken.h"

int main()
{
  // Define the model
  const char *model = "gpt-4";

  CChatCompletionRequestMessage *messages[3];

  messages[0] = tiktoken_chat_message_new("system");
  messages[1] = tiktoken_chat_message_new("user");
  messages[2] = tiktoken_chat_message_new("system");
  if (messages[0] == NULL || messages[1] == NULL || messages[2] == NULL)
  {
    fprintf(stderr, "Failed to create chat messages\n");
    for (size_t i = 0; i < 3; ++i)
    {
      tiktoken_chat_message_destroy(messages[i]);
    }
    return 1;
  }

  tiktoken_chat_message_set_content(messages[0], "You are a helpful assistant that only speaks French.");
  tiktoken_chat_message_set_content(messages[1], "Hello, how are you?");
  tiktoken_chat_message_set_content(messages[2], "Parlez-vous francais?");

  // Get the maximum tokens for chat completion
  size_t max_tokens = tiktoken_get_chat_completion_max_tokens(model, 3, messages);

  // Display the max_token parameter value
  printf("Max_token parameter value: %zu\n", max_tokens);

  for (size_t i = 0; i < 3; ++i)
  {
    tiktoken_chat_message_destroy(messages[i]);
  }

  return 0;
}
