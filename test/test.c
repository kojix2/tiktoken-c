#include "../tiktoken.h"
#include <stdio.h>

int main(int argc, char *argv[]){
    if (argc < 2) {
        printf("Error: No text argument provided.\n");
        return 1;
    }

    char *text = argv[1];
    printf("%s\n", text);

    CoreBPE *bpe = c_get_bpe_from_model("gpt-4");
    size_t n;
    size_t *tokens = c_corebpe_encode_with_special_tokens(bpe, text, &n);

    for(size_t i=0; i<n; i++){
        printf("%zu ", tokens[i]);
    }
    printf("\n");

    return 0;
}
