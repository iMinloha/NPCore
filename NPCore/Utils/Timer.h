#ifndef NPCORE_UTILS_TIMER_H
#define NPCORE_UTILS_TIMER_H

#include "Core/Export.h"

// =================================[Timer 计时工具]================================
// 函数指针类型，用于计时的回调函数

typedef void (*Func)(void *);

// 对函数进行计时分析，输出纳秒级耗时
NPCORE_API void TimeAnalysis(Func func, void *args);

#endif // NPCORE_UTILS_TIMER_H
