#include "Core/RandomGenerator.h"

// 定义静态成员
std::random_device RandomGenerator::seed_source;

// 定义线程本地存储的生成器（自动使用随机设备初始化）
thread_local std::mt19937 RandomGenerator::generator(std::random_device{}());

// 显式模板实例化定义
template float RandomGenerator::uniform<float>(float, float);
template double RandomGenerator::uniform<double>(double, double);
template float RandomGenerator::normal<float>(float, float);
template double RandomGenerator::normal<double>(double, double);
