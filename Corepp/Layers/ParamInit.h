#ifndef COREPP_LAYERS_PARAMINIT_H
#define COREPP_LAYERS_PARAMINIT_H

#include <cmath>
#include "Core/Matrix.hpp"
#include "Core/RandomGenerator.h"
#include "Core/LinearAlgebra.h"

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
                    const InitParams& params = {.a = -1, .b = -1, .mu = 0, .sigma = 1, .value = 1}) {
    switch(mode) {
        case Zeros:
            for(int r = 0; r < mat.row; ++r)
                for(int c = 0; c < mat.col; ++c)
                    for(int ch = 0; ch < mat.channel; ++ch)
                        mat.at(r, c, ch) = 0;
            break;

        case Ones:
            for(int r = 0; r < mat.row; ++r)
                for(int c = 0; c < mat.col; ++c)
                    for(int ch = 0; ch < mat.channel; ++ch)
                        mat.at(r, c, ch) = 1;
            break;

        case Constant:
            for(int r = 0; r < mat.row; ++r)
                for(int c = 0; c < mat.col; ++c)
                    for(int ch = 0; ch < mat.channel; ++ch)
                        mat.at(r, c, ch) = params.value;
            break;

        case Uniform:
            for(int r = 0; r < mat.row; ++r)
                for(int c = 0; c < mat.col; ++c)
                    for(int ch = 0; ch < mat.channel; ++ch)
                        mat.at(r, c, ch) = RandomGenerator::uniform<T>(params.a, params.b);
            break;

        case Gaussian:
            for(int r = 0; r < mat.row; ++r)
                for(int c = 0; c < mat.col; ++c)
                    for(int ch = 0; ch < mat.channel; ++ch)
                        mat.at(r, c, ch) = RandomGenerator::normal<T>(params.mu, params.sigma);
            break;

        case XavierUniform: {
            T limit = std::sqrt(6.0f / (mat.row + mat.col));
            for(int r = 0; r < mat.row; ++r)
                for(int c = 0; c < mat.col; ++c)
                    for(int ch = 0; ch < mat.channel; ++ch)
                        mat.at(r, c, ch) = RandomGenerator::uniform<T>(-limit, limit);
        } break;

        case HeNormal: {
            T stddev = std::sqrt(2.0f / mat.row);
            for(int r = 0; r < mat.row; ++r)
                for(int c = 0; c < mat.col; ++c)
                    for(int ch = 0; ch < mat.channel; ++ch)
                        mat.at(r, c, ch) = RandomGenerator::normal<T>(0, stddev);
        } break;

        case Identity:
            COREPP_ASSERT(mat.row == mat.col, "Identity matrix must be square");
            for(int r = 0; r < mat.row; ++r)
                for(int c = 0; c < mat.col; ++c)
                    for(int ch = 0; ch < mat.channel; ++ch)
                        mat.at(r, c, ch) = (r == c) ? 1 : 0;
            break;

        case Orthogonal: {
            COREPP_ASSERT(mat.row >= mat.col, "Orthogonal init requires rows >= cols");
            const float gain = params.gain;
            InitMatrixFunc(mat, InitMode::Gaussian, {.mu = 0, .sigma = 1});
            GramSchmidtOrthogonalization(mat);
            mat = mat * gain;
        } break;

        default:
            COREPP_ASSERT(true, "Unsupported initialization mode");
    }
}

#endif // COREPP_LAYERS_PARAMINIT_H
