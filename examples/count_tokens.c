#include <stdio.h>
#include <stdlib.h>
#include "tiktoken.h"

int main()
{
  // Initialize the BPE encoder
  CoreBPE *bpe = tiktoken_p50k_base();
  if (bpe == NULL)
  {
    fprintf(stderr, "Failed to initialize BPE\n");
    return 1;
  }

  // Encode the text
  const char *text = "This is a test         with a lot of spaces";
  size_t num_tokens;
  Rank *tokens = tiktoken_corebpe_encode_with_special_tokens(bpe, text, &num_tokens);

  // Display the token count
  if (tokens != NULL)
  {
    printf("Token count: %zu\n", num_tokens);
    free(tokens);
  }
  else
  {
    fprintf(stderr, "Failed to encode text\n");
    tiktoken_destroy_corebpe(bpe);
    return 1;
  }

  // Free the BPE encoder
  tiktoken_destroy_corebpe(bpe);
  return 0;
}
