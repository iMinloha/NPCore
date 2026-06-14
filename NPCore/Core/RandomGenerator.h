#ifndef NPCORE_CORE_RANDOMGENERATOR_H
#define NPCORE_CORE_RANDOMGENERATOR_H

#include <random>
#include <mutex>

class RandomGenerator {
private:
    static std::random_device seed_source;

    // 线程本地存储的生成器
    static thread_local std::mt19937 generator;

    // 禁用构造函数
    RandomGenerator() = delete;

public:
    // 初始化种子（线程安全）
    static void init_seed(unsigned int seed = 0);

    // 均匀分布
    template<typename T>
    static T uniform(T min, T max);

    // 高斯分布
    template<typename T>
    static T normal(T mean, T stddev);
};

#endif // NPCORE_CORE_RANDOMGENERATOR_H
