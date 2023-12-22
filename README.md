# tiktoken-c

This library provides an unofficial C API for [Tiktoken](https://github.com/openai/tiktoken).

[![test](https://github.com/kojix2/tiktoken-c/actions/workflows/test.yml/badge.svg)](https://github.com/kojix2/tiktoken-c/actions/workflows/test.yml)

- This library adds a simple API for C to [tiktoken-rs](https://github.com/zurawiki/tiktoken-rs).
- This library was created for [tiktoken-cr](https://github.com/kojix2/tiktoken-cr).

## Build

```sh
git clone https://github.com/kojix2/tiktoken-c
cd tiktoken-c
# Create shared library
cargo build --release
# target/release/libtiktoken_c.so
```

## API
    
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

## cbindgen

```
cargo install --force cbindgen
cbindgen --config cbindgen.toml --crate tiktoken-c --output tiktoken.h
# Add Opaque Pointer
perl -i -pe '$i ||= /#include/; $_ = "\ntypedef void CoreBPE;\n" if $i && /^$/ && !$f++; $i = 0 if /^$/ && $f' tiktoken.h
```

## Contributing

- Report bugs
- Fix bugs and submit pull requests
- Write, clarify, or fix documentation
- Suggest or add new features
- Make a donation

# License

MIT
