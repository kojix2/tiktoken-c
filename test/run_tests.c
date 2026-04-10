#include "../tiktoken.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct TestCase {
    const char *model;
    const char *text;
    const Rank *expected_tokens;
    size_t expected_len;
} TestCase;

typedef struct ModelMetadataCase {
    const char *model;
    size_t expected_context_size;
    TiktokenTokenizer expected_tokenizer;
} ModelMetadataCase;

static int print_tokens(const Rank *tokens, size_t len)
{
    size_t i;

    for (i = 0; i < len; i++)
    {
        if (i > 0)
        {
            printf(" ");
        }
        printf("%u", tokens[i]);
    }

    return 0;
}

static int run_test_case(const TestCase *test_case)
{
    CoreBPE *bpe;
    Rank *tokens;
    size_t token_count;
    size_t i;

    bpe = tiktoken_get_bpe_from_model(test_case->model);
    if (bpe == NULL)
    {
        fprintf(stderr, "Failed to load model: %s\n", test_case->model);
        return 1;
    }

    tokens = tiktoken_corebpe_encode_with_special_tokens(bpe, test_case->text, &token_count);
    if (tokens == NULL && test_case->expected_len != 0)
    {
        fprintf(stderr, "Encoding failed for model: %s\n", test_case->model);
        tiktoken_destroy_corebpe(bpe);
        return 1;
    }

    if (token_count != test_case->expected_len)
    {
        fprintf(stderr, "Length mismatch for %s\n", test_case->model);
        fprintf(stderr, "Expected: ");
        print_tokens(test_case->expected_tokens, test_case->expected_len);
        fprintf(stderr, "\nGot: ");
        print_tokens(tokens, token_count);
        fprintf(stderr, "\n");
        tiktoken_free(tokens);
        tiktoken_destroy_corebpe(bpe);
        return 1;
    }

    for (i = 0; i < token_count; i++)
    {
        if (tokens[i] != test_case->expected_tokens[i])
        {
            fprintf(stderr, "Token mismatch for %s at index %zu\n", test_case->model, i);
            fprintf(stderr, "Expected: ");
            print_tokens(test_case->expected_tokens, test_case->expected_len);
            fprintf(stderr, "\nGot: ");
            print_tokens(tokens, token_count);
            fprintf(stderr, "\n");
            tiktoken_free(tokens);
            tiktoken_destroy_corebpe(bpe);
            return 1;
        }
    }

    printf("Testing %s\n", test_case->model);
    printf("Expected: ");
    print_tokens(test_case->expected_tokens, test_case->expected_len);
    printf("\nGot: ");
    print_tokens(tokens, token_count);
    printf("\nTest passed successfully for %s\n", test_case->model);

    tiktoken_free(tokens);
    tiktoken_destroy_corebpe(bpe);
    return 0;
}

int main(void)
{
    static const Rank gpt4_tokens[] = {40, 1097, 264, 8415, 13};
    static const Rank o200k_tokens[] = {40, 939, 261, 9059, 13};
    static const TestCase test_cases[] = {
        {"gpt-4", "I am a cat.", gpt4_tokens, sizeof(gpt4_tokens) / sizeof(gpt4_tokens[0])},
        {"gpt-4o", "I am a cat.", o200k_tokens, sizeof(o200k_tokens) / sizeof(o200k_tokens[0])},
        {"gpt-5", "I am a cat.", o200k_tokens, sizeof(o200k_tokens) / sizeof(o200k_tokens[0])},
        {"gpt-oss-20b", "I am a cat.", o200k_tokens, sizeof(o200k_tokens) / sizeof(o200k_tokens[0])},
        {"gpt-oss-120b", "I am a cat.", o200k_tokens, sizeof(o200k_tokens) / sizeof(o200k_tokens[0])},
    };
    static const ModelMetadataCase metadata_cases[] = {
        {"gpt-4", 8192, TIKTOKEN_TOKENIZER_CL100K_BASE},
        {"gpt-4o", 128000, TIKTOKEN_TOKENIZER_O200K_BASE},
        {"gpt-5", 400000, TIKTOKEN_TOKENIZER_O200K_BASE},
        {"gpt2", 0, TIKTOKEN_TOKENIZER_GPT2},
        {"gpt-oss-20b", 131072, TIKTOKEN_TOKENIZER_O200K_HARMONY},
    };
    size_t i;

    printf("tiktoken_c version: %s\n", tiktoken_c_version());

    for (i = 0; i < sizeof(metadata_cases) / sizeof(metadata_cases[0]); i++)
    {
        const ModelMetadataCase *test_case = &metadata_cases[i];
        TiktokenTokenizer tokenizer = tiktoken_get_tokenizer(test_case->model);

        if (tokenizer != test_case->expected_tokenizer)
        {
            fprintf(stderr, "Tokenizer mismatch for %s: expected %d, got %d\n",
                    test_case->model,
                    (int)test_case->expected_tokenizer,
                    (int)tokenizer);
            return 1;
        }

        if (test_case->expected_context_size != 0)
        {
            size_t context_size = tiktoken_get_context_size(test_case->model);
            if (context_size != test_case->expected_context_size)
            {
                fprintf(stderr, "Context size mismatch for %s: expected %zu, got %zu\n",
                        test_case->model,
                        test_case->expected_context_size,
                        context_size);
                return 1;
            }
        }
    }

    for (i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++)
    {
        if (run_test_case(&test_cases[i]) != 0)
        {
            return 1;
        }
    }

    printf("All tests passed successfully\n");
    return 0;
}