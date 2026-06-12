#include "Utils/Timer.h"
#include <chrono>
#include <iostream>

void TimeAnalysis(Func func, void *args) {
    auto beforeTime = std::chrono::steady_clock::now();
    func(args);
    auto afterTime = std::chrono::steady_clock::now();
    double duration_nanosecond = std::chrono::duration<double, std::nano>(afterTime - beforeTime).count();
    std::cout << duration_nanosecond << "ns" << std::endl;
}
