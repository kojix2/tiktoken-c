#include <stdio.h>
#include <string.h>

// Include twice to verify that the public header is guarded.
#include <tiktoken.h>
#include <tiktoken.h>

int main(void)
{
    const char *version = tiktoken_c_version();

#ifdef TIKTOKEN_C_ENABLE_LOGGING
    tiktoken_init_logger();
#endif

    if (version == NULL || strlen(version) == 0)
    {
        fprintf(stderr, "Unexpected tiktoken-c version: %s\n", version ? version : "NULL");
        return 1;
    }

    return 0;
}
