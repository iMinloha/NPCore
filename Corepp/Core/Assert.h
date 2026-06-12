#ifndef COREPP_CORE_ASSERT_H
#define COREPP_CORE_ASSERT_H

#include <cstdio>
#include <cstdlib>

#define COREPP_ASSERT(condition, ...) \
        if (!(condition)) { \
            printf("CorePP Assert: "); \
            printf(__VA_ARGS__); \
            printf("\n"); \
            exit(1); \
        }

#endif // COREPP_CORE_ASSERT_H
