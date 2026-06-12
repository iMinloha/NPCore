#ifndef COREPP_UTILS_TIMER_H
#define COREPP_UTILS_TIMER_H

// =================================[Timer 计时工具]================================
// 函数指针类型，用于计时的回调函数

typedef void (*Func)(void *);

// 对函数进行计时分析，输出纳秒级耗时
void TimeAnalysis(Func func, void *args);

#endif // COREPP_UTILS_TIMER_H
