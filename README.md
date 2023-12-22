# tiktoken-c

[![test](https://github.com/kojix2/tiktoken-c/actions/workflows/test.yml/badge.svg)](https://github.com/kojix2/tiktoken-c/actions/workflows/test.yml)

- This library provides an unofficial C API for [Tiktoken](https://github.com/openai/tiktoken).
- It allows tiktoken to be used from a variety of programming languages.
- This library adds a simple API for C to [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs).

## Build

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

void c_init_logger(void);

CoreBPE *c_r50k_base(void);

CoreBPE *c_p50k_base(void);

CoreBPE *c_p50k_edit(void);

CoreBPE *c_cl100k_base(void);

void c_destroy_corebpe(CoreBPE *ptr);

CoreBPE *c_get_bpe_from_model(const char *model);

size_t c_get_completion_max_tokens(const char *model, const char *prompt);

size_t c_num_tokens_from_messages(const char *model,
                                  uint32_t num_messages,
                                  const struct CChatCompletionRequestMessage *messages);

size_t c_get_chat_completion_max_tokens(const char *model,
                                        uint32_t num_messages,
                                        const struct CChatCompletionRequestMessage *messages);

size_t *c_corebpe_encode_ordinary(CoreBPE *ptr, const char *text, size_t *num_tokens);

size_t *c_corebpe_encode(CoreBPE *ptr,
                         const char *text,
                         const char *const *allowed_special,
                         size_t allowed_special_len,
                         size_t *num_tokens);

size_t *c_corebpe_encode_with_special_tokens(CoreBPE *ptr, const char *text, size_t *num_tokens);

char *c_corebpe_decode(CoreBPE *ptr, const size_t *tokens, size_t num_tokens);
```

## Language Bindings

|Language|Bindings|
|---|---|
|Crystal|[tiktoken-cr](https://github.com/kojix2/tiktoken-cr)|

## Development

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
perl -i -pe '$i ||= /#include/; $_ = "\ntypedef void CoreBPE;\n" if $i && /^$/ && !$f++; $i = 0 if /^$/ && $f' tiktoken.h
```

# License

MIT
