/* https://github.com/kojix2/tiktoken-c */

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif

  typedef void CoreBPE;
  typedef uint32_t Rank;

  typedef struct CChatCompletionRequestMessage CChatCompletionRequestMessage;

  const char *tiktoken_c_version(void);

  void tiktoken_init_logger(void);

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

  bool tiktoken_chat_message_set_refusal(CChatCompletionRequestMessage *message,
                                         const char *refusal);

  void tiktoken_chat_message_destroy(CChatCompletionRequestMessage *message);

  CoreBPE *tiktoken_get_bpe_from_model(const char *model);

  CoreBPE *tiktoken_r50k_base(void);

  CoreBPE *tiktoken_p50k_base(void);

  CoreBPE *tiktoken_p50k_edit(void);

  CoreBPE *tiktoken_cl100k_base(void);

  CoreBPE *tiktoken_o200k_base(void);

  CoreBPE *tiktoken_o200k_harmony(void);

  Rank *tiktoken_corebpe_encode_ordinary(CoreBPE *ptr, const char *text, size_t *num_tokens);

  size_t tiktoken_corebpe_count_ordinary(CoreBPE *ptr, const char *text);

  Rank *tiktoken_corebpe_encode(CoreBPE *ptr,
                                const char *text,
                                const char *const *allowed_special,
                                size_t allowed_special_len,
                                size_t *num_tokens);

  size_t tiktoken_corebpe_count(CoreBPE *ptr,
                                const char *text,
                                const char *const *allowed_special,
                                size_t allowed_special_len);

  Rank *tiktoken_corebpe_encode_with_special_tokens(CoreBPE *ptr,
                                                    const char *text,
                                                    size_t *num_tokens);

  size_t tiktoken_corebpe_count_with_special_tokens(CoreBPE *ptr, const char *text);

  char *tiktoken_corebpe_decode(CoreBPE *ptr, const Rank *tokens, size_t num_tokens);

  uint8_t *tiktoken_corebpe_decode_bytes(CoreBPE *ptr,
                                         const Rank *tokens,
                                         size_t num_tokens,
                                         size_t *num_bytes);

  size_t tiktoken_get_completion_max_tokens(const char *model, const char *prompt);

  size_t tiktoken_num_tokens_from_messages(const char *model,
                                           uint32_t num_messages,
                 CChatCompletionRequestMessage *const *messages);

  size_t tiktoken_get_chat_completion_max_tokens(const char *model,
                                                 uint32_t num_messages,
                   CChatCompletionRequestMessage *const *messages);

  void tiktoken_free(void *ptr);

  void tiktoken_destroy_corebpe(CoreBPE *ptr);

#ifdef __cplusplus
}
#endif