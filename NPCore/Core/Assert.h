#ifndef NPCORE_CORE_ASSERT_H
#define NPCORE_CORE_ASSERT_H

#include <cstdio>
#include <cstdlib>

#define NPCORE_ASSERT(condition, ...) \
        if (!(condition)) { \
            printf("NPCore Assert: "); \
            printf(__VA_ARGS__); \
            printf("\n"); \
            exit(1); \
        }

#endif // NPCORE_CORE_ASSERT_H
