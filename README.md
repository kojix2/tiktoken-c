# tiktoken-c

[![test](https://github.com/kojix2/tiktoken-c/actions/workflows/test.yml/badge.svg)](https://github.com/kojix2/tiktoken-c/actions/workflows/test.yml)
[![Lines of Code](https://img.shields.io/endpoint?url=https%3A%2F%2Ftokei.kojix2.net%2Fbadge%2Fgithub%2Fkojix2%2Ftiktoken-c%2Flines)](https://tokei.kojix2.net/github/kojix2/tiktoken-c)

- C API for [Tiktoken](https://github.com/openai/tiktoken), OpenAI's tokenizer
- Compatible with [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs) 0.7.0+

## Installation

Download from [GitHub Releases](https://github.com/kojix2/tiktoken-c/releases) or build from source:

```sh
git clone https://github.com/kojix2/tiktoken-c
cd tiktoken-c
cargo build --release
# Output: target/release/libtiktoken_c.so (Linux), .dylib (macOS), or .dll (Windows)
```

## API

See [tiktoken-rs documentation](https://docs.rs/tiktoken-rs/) for detailed behavior.

### Memory Management

- Free memory from `tiktoken_corebpe_decode` and `Rank*` arrays with `free()`
- Free `CoreBPE*` objects with `tiktoken_destroy_corebpe`

### Error Handling

- Functions return `NULL` or `usize::MAX` on error
- Model names must match those supported by tiktoken-rs

```c
typedef void CoreBPE;
typedef uint32_t Rank;

typedef struct CFunctionCall {
  const char *name;
  const char *arguments;
} CFunctionCall;

typedef struct CChatCompletionRequestMessage {
  const char *role;
  const char *content;
  const char *name;
  const struct CFunctionCall *function_call;
} CChatCompletionRequestMessage;

// Library version
const char *tiktoken_c_version(void);

// Initialize logger
void tiktoken_init_logger(void);

// Get tokenizers
CoreBPE *tiktoken_r50k_base(void);     // GPT-3 models
CoreBPE *tiktoken_p50k_base(void);     // Code models, text-davinci-002/003
CoreBPE *tiktoken_p50k_edit(void);     // Edit models
CoreBPE *tiktoken_cl100k_base(void);   // ChatGPT models, embeddings
CoreBPE *tiktoken_o200k_base(void);    // GPT-4o models

// Free a CoreBPE instance
void tiktoken_destroy_corebpe(CoreBPE *ptr);

// Get tokenizer for a specific model
CoreBPE *tiktoken_get_bpe_from_model(const char *model);

// Token calculations
size_t tiktoken_get_completion_max_tokens(const char *model, const char *prompt);
size_t tiktoken_num_tokens_from_messages(const char *model,
                                         uint32_t num_messages,
                                         const struct CChatCompletionRequestMessage *messages);
size_t tiktoken_get_chat_completion_max_tokens(const char *model,
                                               uint32_t num_messages,
                                               const struct CChatCompletionRequestMessage *messages);

// Encoding/decoding
Rank *tiktoken_corebpe_encode_ordinary(CoreBPE *ptr, const char *text, size_t *num_tokens);
Rank *tiktoken_corebpe_encode(CoreBPE *ptr,
                              const char *text,
                              const char *const *allowed_special,
                              size_t allowed_special_len,
                              size_t *num_tokens);
Rank *tiktoken_corebpe_encode_with_special_tokens(CoreBPE *ptr,
                                                  const char *text,
                                                  size_t *num_tokens);
char *tiktoken_corebpe_decode(CoreBPE *ptr, const Rank *tokens, size_t num_tokens);
```

## Example: Counting Tokens

```c
#include <stdio.h>
#include <stdlib.h>
#include "tiktoken.h"

int main() {
  CoreBPE *bpe = tiktoken_get_bpe_from_model("gpt-4");
  if (!bpe) return 1;

  const char *text = "Hello, world!";
  size_t num_tokens;
  Rank *tokens = tiktoken_corebpe_encode_with_special_tokens(bpe, text, &num_tokens);

  if (tokens) {
    printf("Token count: %zu\n", num_tokens);
    free(tokens);
  }

  tiktoken_destroy_corebpe(bpe);
  return 0;
}
```

## Language Bindings

| Language | Bindings                                             |
| -------- | ---------------------------------------------------- |
| Crystal  | [tiktoken-cr](https://github.com/kojix2/tiktoken-cr) |

## Development

```sh
# Run tests
cargo test
cd test && ./test.sh

# Generate header file
cargo install --force cbindgen
cbindgen --config cbindgen.toml --crate tiktoken-c --output tiktoken.h
perl -i -pe '$i ||= /#include/; $_ = "\ntypedef void CoreBPE;\ntypedef uint32_t Rank;\n" if $i && /^$/ && !$f++; $i = 0 if /^$/ && $f' tiktoken.h
```

## License

MIT
