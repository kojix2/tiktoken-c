# tiktoken-c

[![test](https://github.com/kojix2/tiktoken-c/actions/workflows/test.yml/badge.svg)](https://github.com/kojix2/tiktoken-c/actions/workflows/test.yml)

- This library provides an unofficial C API for [Tiktoken](https://github.com/openai/tiktoken).
- It allows tiktoken to be used from a variety of programming languages.
- This library adds a simple API for C to [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs).

## Installation

Download binaries from [GitHub Releases](https://github.com/kojix2/tiktoken-c/releases).

From source code:

```sh
git clone https://github.com/kojix2/tiktoken-c
cd tiktoken-c
# Create shared library
cargo build --release
# target/release/libtiktoken_c.so
```

## API

Please refer to the [tiktoken-rs documentation](https://docs.rs/tiktoken-rs/).

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

const char *tiktoken_c_version(void);

void tiktoken_init_logger(void);

CoreBPE *tiktoken_r50k_base(void);

CoreBPE *tiktoken_p50k_base(void);

CoreBPE *tiktoken_p50k_edit(void);

CoreBPE *tiktoken_cl100k_base(void);

CoreBPE *tiktoken_o200k_base(void);

void tiktoken_destroy_corebpe(CoreBPE *ptr);

CoreBPE *tiktoken_get_bpe_from_model(const char *model);

size_t tiktoken_get_completion_max_tokens(const char *model, const char *prompt);

size_t tiktoken_num_tokens_from_messages(const char *model,
                                         uint32_t num_messages,
                                         const struct CChatCompletionRequestMessage *messages);

size_t tiktoken_get_chat_completion_max_tokens(const char *model,
                                               uint32_t num_messages,
                                               const struct CChatCompletionRequestMessage *messages);

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

## Language Bindings

| Language | Bindings                                             |
| -------- | ---------------------------------------------------- |
| Crystal  | [tiktoken-cr](https://github.com/kojix2/tiktoken-cr) |

## Development

The code for this project was created by fully utilizing ChatGPT and GitHub Copilot.

Run tests

```
# Tests written in Rust
cargo test
# Tests in C
cd test && ./test.sh
```

Create header files with [cbindgen](https://github.com/mozilla/cbindgen)

```
cargo install --force cbindgen
cbindgen --config cbindgen.toml --crate tiktoken-c --output tiktoken.h
```

cbindgen does not support opaque pointers and must be added.

```
perl -i -pe '$i ||= /#include/; $_ = "\ntypedef void CoreBPE;\ntypedef uint32_t Rank;\n" if $i && /^$/ && !$f++; $i = 0 if /^$/ && $f' tiktoken.h
```

## License

MIT
