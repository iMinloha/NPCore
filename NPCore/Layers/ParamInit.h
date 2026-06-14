#ifndef NPCORE_LAYERS_PARAMINIT_H
#define NPCORE_LAYERS_PARAMINIT_H

#include "Core/Matrix.hpp"

// =================================[参数初始化方法]================================

typedef enum {
    Zeros,          // 全部初始化为0
    Ones,           // 全部初始化为1
    Constant,       // 全部初始化为常数
    Uniform,        // 均匀分布
    Gaussian,       // 高斯分布
    XavierUniform,  // Xavier均匀分布
    HeNormal,       // He高斯分布
    Identity,       // 单位矩阵
    Orthogonal      // 正交矩阵
} InitMode;

struct InitParams {
    double a = -1;      // Uniform下限
    double b = 1;       // Uniform上限
    double mu = 0;      // 高斯均值
    double sigma = 1;   // 高斯标准差
    double value = 1;   // 常数值
    float gain = 1.0f;  // Orthogonal增益参数
};

template<typename T>
void InitMatrixFunc(Matrix<T>& mat, InitMode mode,
                    const InitParams& params = {.a = -1, .b = -1, .mu = 0, .sigma = 1, .value = 1});

#endif // NPCORE_LAYERS_PARAMINIT_H
