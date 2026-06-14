#include "Core/RandomGenerator.h"

// 定义静态成员
std::random_device RandomGenerator::seed_source;

// 定义线程本地存储的生成器（自动使用随机设备初始化）
thread_local std::mt19937 RandomGenerator::generator(std::random_device{}());

// 初始化种子（线程安全）
void RandomGenerator::init_seed(unsigned int seed) {
    static std::once_flag flag;
    std::call_once(flag, [seed]() {
        if (seed == 0) {
            generator.seed(seed_source());
        } else {
            generator.seed(seed);
        }
    });
}

// 均匀分布
template<typename T>
T RandomGenerator::uniform(T min, T max) {
    std::uniform_real_distribution<T> dist(min, max);
    return dist(generator);
}

// 高斯分布
template<typename T>
T RandomGenerator::normal(T mean, T stddev) {
    std::normal_distribution<T> dist(mean, stddev);
    return dist(generator);
}

// 显式模板实例化定义
template float RandomGenerator::uniform<float>(float, float);
template double RandomGenerator::uniform<double>(double, double);
template float RandomGenerator::normal<float>(float, float);
template double RandomGenerator::normal<double>(double, double);
