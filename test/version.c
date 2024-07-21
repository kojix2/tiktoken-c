#include "../tiktoken.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
    printf("tiktoken_c version: %s\n", tiktoken_c_version());
    return 0;
}