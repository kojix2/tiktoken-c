# tiktoken-c

[![build](https://github.com/kojix2/tiktoken-c/actions/workflows/build.yml/badge.svg)](https://github.com/kojix2/tiktoken-c/actions/workflows/build.yml)
[![Lines of Code](https://img.shields.io/endpoint?url=https%3A%2F%2Ftokei.kojix2.net%2Fbadge%2Fgithub%2Fkojix2%2Ftiktoken-c%2Flines)](https://tokei.kojix2.net/github/kojix2/tiktoken-c)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kojix2/tiktoken-c)

- C API for [Tiktoken](https://github.com/openai/tiktoken), OpenAI's tokenizer
- Compatible with [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs) 0.11.0
- This project prioritizes tracking upstream tiktoken-rs; C API and ABI may change between releases.

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
typedef enum TiktokenTokenizer TiktokenTokenizer;
typedef struct CChatCompletionRequestMessage CChatCompletionRequestMessage;
```

### Core Functions

#### Version / Init

```c
const char *tiktoken_c_version(void);
void tiktoken_init_logger(void);
size_t tiktoken_get_context_size(const char *model);
TiktokenTokenizer tiktoken_get_tokenizer(const char *model);

CChatCompletionRequestMessage *tiktoken_chat_message_new(const char *role);
bool tiktoken_chat_message_set_role(CChatCompletionRequestMessage *message, const char *role);
bool tiktoken_chat_message_set_content(CChatCompletionRequestMessage *message, const char *content);
bool tiktoken_chat_message_set_name(CChatCompletionRequestMessage *message, const char *name);
bool tiktoken_chat_message_set_function_call(CChatCompletionRequestMessage *message,
                                             const char *name,
                                             const char *arguments);
void tiktoken_chat_message_clear_function_call(CChatCompletionRequestMessage *message);
bool tiktoken_chat_message_add_tool_call(CChatCompletionRequestMessage *message,
                                         const char *name,
                                         const char *arguments);
void tiktoken_chat_message_clear_tool_calls(CChatCompletionRequestMessage *message);
bool tiktoken_chat_message_set_refusal(CChatCompletionRequestMessage *message, const char *refusal);
void tiktoken_chat_message_destroy(CChatCompletionRequestMessage *message);
```

#### Load Tokenizer

```c
CoreBPE *tiktoken_get_bpe_from_model(const char *model);
CoreBPE *tiktoken_r50k_base(void);     // GPT-3 models
CoreBPE *tiktoken_p50k_base(void);     // Code models
CoreBPE *tiktoken_p50k_edit(void);     // Edit models
CoreBPE *tiktoken_cl100k_base(void);   // ChatGPT models
CoreBPE *tiktoken_o200k_base(void);    // GPT-5, GPT-4.1, GPT-4o, o4, o3, and o1 models
CoreBPE *tiktoken_o200k_harmony(void); // gpt-oss models, gpt-oss-20b, gpt-oss-120b
```

#### Encoding & Decoding

```c
Rank *tiktoken_corebpe_encode(CoreBPE *ptr, const char *text,
                              const char *const *allowed_special,
                              size_t allowed_special_len,
                              size_t *num_tokens);

Rank *tiktoken_corebpe_encode_ordinary(CoreBPE *ptr, const char *text, size_t *num_tokens);
size_t tiktoken_corebpe_count_ordinary(CoreBPE *ptr, const char *text);
size_t tiktoken_corebpe_count(CoreBPE *ptr, const char *text,
                              const char *const *allowed_special,
                              size_t allowed_special_len);
Rank *tiktoken_corebpe_encode_with_special_tokens(CoreBPE *ptr, const char *text, size_t *num_tokens);
size_t tiktoken_corebpe_count_with_special_tokens(CoreBPE *ptr, const char *text);
char *tiktoken_corebpe_decode(CoreBPE *ptr, const Rank *tokens, size_t num_tokens);
uint8_t *tiktoken_corebpe_decode_bytes(CoreBPE *ptr, const Rank *tokens,
                                       size_t num_tokens, size_t *num_bytes);
```

#### Token Counting

```c
size_t tiktoken_get_text_completion_max_tokens(const char *model, const char *prompt);

size_t tiktoken_num_tokens_from_messages(const char *model,
                                         uint32_t num_messages,
                 CChatCompletionRequestMessage *const *messages);

size_t tiktoken_get_chat_completion_max_tokens(const char *model,
                                               uint32_t num_messages,
                   CChatCompletionRequestMessage *const *messages);
```

#### Cleanup

```c
void tiktoken_destroy_corebpe(CoreBPE *ptr);
void tiktoken_free(void *ptr);
```

## Memory Management

Use `tiktoken_free()` to release any heap memory returned by the library:

| Function                                              | Return Type       | Free with                    |
| ----------------------------------------------------- | ----------------- | ---------------------------- |
| `*_encode*` / `*_decode*`                             | `Rank*` / `char*` / `uint8_t*` | `tiktoken_free(ptr)`         |
| `tiktoken_*_base()` / `tiktoken_get_bpe_from_model()` | `CoreBPE*`        | `tiktoken_destroy_corebpe()` |
| `tiktoken_chat_message_new()`                         | `CChatCompletionRequestMessage*` | `tiktoken_chat_message_destroy()` |

The `*_count*` APIs return `size_t` directly and do not allocate memory.

Important Notes:

- Do NOT pass the pointer returned by `tiktoken_c_version()` to any free function (static string).
- On Windows, always prefer `tiktoken_free()` rather than `free()`.
- When encoding results in 0 tokens, the returned pointer may be NULL. Always check for NULL before use.

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
    tiktoken_free(tokens);
  }

  tiktoken_destroy_corebpe(bpe);
  return 0;
}
```

### Count Chat Tokens

```c
#include <stdio.h>
#include "tiktoken.h"

int main() {
  CChatCompletionRequestMessage *messages[2];
  messages[0] = tiktoken_chat_message_new("assistant");
  messages[1] = tiktoken_chat_message_new("assistant");
  if (messages[0] == NULL || messages[1] == NULL) return 1;

  tiktoken_chat_message_set_content(messages[0], "I'll call the weather tool.");
  tiktoken_chat_message_add_tool_call(messages[0], "get_weather", "{\"location\":\"Tokyo\"}");
  tiktoken_chat_message_set_refusal(messages[1], "I cannot help with that request.");

  size_t num_tokens = tiktoken_num_tokens_from_messages("gpt-4o", 2, messages);
  printf("Chat tokens: %zu\n", num_tokens);

  tiktoken_chat_message_destroy(messages[0]);
  tiktoken_chat_message_destroy(messages[1]);
  return 0;
}
```

## Language Bindings

| Language | Repository                                           |
| -------- | ---------------------------------------------------- |
| Crystal  | [tiktoken-cr](https://github.com/kojix2/tiktoken-cr) |

## Development

tiktoken-c prioritizes tracking upstream [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs) over preserving a stable C ABI across releases. When upstream changes require new fields, new layouts, or API reshaping, this project may make breaking changes to C structs and function signatures instead of keeping compatibility shims indefinitely. If you maintain a downstream binding or embed the header directly, treat new releases as potentially ABI-breaking and recompile against the matching version of [tiktoken.h](tiktoken.h).

The C integration test logic is centralized in [test/run_tests.c](test/run_tests.c) and is intended to run with the same expectations on Linux, macOS, and Windows.

```sh
# Run tests
cargo test
cd test && ./test.sh

# Test with release build
cd test && BUILD_DIR=release ./test.sh
```

## License

MIT
