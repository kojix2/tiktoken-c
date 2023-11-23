#include "../tiktoken.h"
#include <stdio.h>
#include <string.h>
#include <sys/types.h>

int main(void)
{
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

    CoreBPE *bpe = c_get_bpe_from_model("gpt-4");
    size_t n;
    size_t *tokens = c_corebpe_encode_with_special_tokens(bpe, text, &n);

    for (size_t i = 0; i < n; i++)
    {
        printf("%zu", tokens[i]);
        if (i < n - 1)
        {
            printf(" ");
        }
    }

    free(text); // do not forget to free allocated memory
    free(tokens);

    return 0;
}
