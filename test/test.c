#include "../tiktoken.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/types.h>

int main(int argc, char *argv[])
{
    char *model = "gpt-4"; // default model
    int opt;

    while ((opt = getopt(argc, argv, "m:")) != -1)
    {
        switch (opt)
        {
        case 'm':
            model = optarg;
            break;
        default:
            fprintf(stderr, "Usage: %s [-m model]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    char *text = NULL;
    size_t len = 0;
    ssize_t nread;

    nread = getline(&text, &len, stdin);
    if (nread == -1)
    {
        printf("Error: No text received from stdin or reading error.\n");
        return 1;
    }

    // getline retains newline character. We may want to remove this.
    if (nread > 0 && text[nread - 1] == '\n')
    {
        text[--nread] = '\0';
    }

    CoreBPE *bpe = tiktoken_get_bpe_from_model(model);
    size_t n;
    Rank *tokens = tiktoken_corebpe_encode_with_special_tokens(bpe, text, &n);

    for (size_t i = 0; i < n; i++)
    {
        printf("%u", tokens[i]);
        if (i < n - 1)
        {
            printf(" ");
        }
    }

    free(text); // do not forget to free allocated memory
    free(tokens);
    tiktoken_destroy_corebpe(bpe); // do not forget to free CoreBPE instance

    return 0;
}
