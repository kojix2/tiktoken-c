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
# Output: target/release/libtiktoken_c.{so,dylib,dll}
```

## C API Overview

The API mirrors the functionality of [tiktoken-rs](https://docs.rs/tiktoken-rs/). Below are key types and functions.

### Types

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
```

### Core Functions

#### Version / Init

```c
const char *tiktoken_c_version(void);
void tiktoken_init_logger(void);
```

#### Load Tokenizer

```c
CoreBPE *tiktoken_get_bpe_from_model(const char *model);
CoreBPE *tiktoken_r50k_base(void);   // GPT-3 models
CoreBPE *tiktoken_p50k_base(void);   // Code models
CoreBPE *tiktoken_p50k_edit(void);   // Edit models
CoreBPE *tiktoken_cl100k_base(void); // ChatGPT models
CoreBPE *tiktoken_o200k_base(void);  // GPT-4o models
```

#### Encoding & Decoding

```c
Rank *tiktoken_corebpe_encode(CoreBPE *ptr, const char *text,
                              const char *const *allowed_special,
                              size_t allowed_special_len,
                              size_t *num_tokens);

Rank *tiktoken_corebpe_encode_ordinary(CoreBPE *ptr, const char *text, size_t *num_tokens);
Rank *tiktoken_corebpe_encode_with_special_tokens(CoreBPE *ptr, const char *text, size_t *num_tokens);
char *tiktoken_corebpe_decode(CoreBPE *ptr, const Rank *tokens, size_t num_tokens);
```

#### Token Counting

```c
size_t tiktoken_get_completion_max_tokens(const char *model, const char *prompt);

size_t tiktoken_num_tokens_from_messages(const char *model,
                                         uint32_t num_messages,
                                         const CChatCompletionRequestMessage *messages);

size_t tiktoken_get_chat_completion_max_tokens(const char *model,
                                               uint32_t num_messages,
                                               const CChatCompletionRequestMessage *messages);
```

#### Cleanup

```c
void tiktoken_destroy_corebpe(CoreBPE *ptr);
```

## Memory Management

Be sure to free memory returned from the API appropriately:

| Function                                              | Return Type       | Free with                    |
| ----------------------------------------------------- | ----------------- | ---------------------------- |
| `*_encode*` / `*_decode`                              | `Rank*` / `char*` | `free()`                     |
| `tiktoken_*_base()` / `tiktoken_get_bpe_from_model()` | `CoreBPE*`        | `tiktoken_destroy_corebpe()` |

**Important Notes:**

- Never use `free()` on a `CoreBPE*`; use `tiktoken_destroy_corebpe()`.
- Always `free()` the result of `encode`/`decode`.

## Example

### Count Tokens

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

| Language | Repository                                           |
| -------- | ---------------------------------------------------- |
| Crystal  | [tiktoken-cr](https://github.com/kojix2/tiktoken-cr) |

## Development

```sh
# Run tests
cargo test
cd test && ./test.sh

# Generate header
cargo install --force cbindgen
cbindgen --config cbindgen.toml --crate tiktoken-c --output tiktoken.h

# Patch header to insert typedefs for CoreBPE and Rank
perl -i -pe '$i ||= /#include/; $_ = "\ntypedef void CoreBPE;\ntypedef uint32_t Rank;\n" if $i && /^$/ && !$f++; $i = 0 if /^$/ && $f' tiktoken.h
```

## License

MIT
