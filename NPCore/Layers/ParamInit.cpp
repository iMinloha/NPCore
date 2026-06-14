#include "Layers/ParamInit.h"
#include <cmath>
#include "Core/RandomGenerator.h"
#include "Core/LinearAlgebra.h"
#include "Core/Assert.h"

template<typename T>
void InitMatrixFunc(Matrix<T>& mat, InitMode mode,
                    const InitParams& params) {
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
            NPCORE_ASSERT(mat.row == mat.col, "Identity matrix must be square");
            for(int r = 0; r < mat.row; ++r)
                for(int c = 0; c < mat.col; ++c)
                    for(int ch = 0; ch < mat.channel; ++ch)
                        mat.at(r, c, ch) = (r == c) ? 1 : 0;
            break;

        case Orthogonal: {
            NPCORE_ASSERT(mat.row >= mat.col, "Orthogonal init requires rows >= cols");
            const float gain = params.gain;
            InitMatrixFunc(mat, InitMode::Gaussian, {.mu = 0, .sigma = 1});
            GramSchmidtOrthogonalization(mat);
            mat = mat * gain;
        } break;

        default:
            NPCORE_ASSERT(true, "Unsupported initialization mode");
    }
}

// Explicit instantiation for float
template void InitMatrixFunc<float>(Matrix<float>&, InitMode, const InitParams&);
