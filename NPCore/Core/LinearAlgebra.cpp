#include "Core/LinearAlgebra.h"
#include <cmath>
#include <vector>
#include <stdexcept>

// =================================[Gram-Schmidt 正交化]================================
template<typename T>
void GramSchmidtOrthogonalization(Matrix<T>& mat) {
    const int rows = mat.row;
    const int cols = mat.col;

    std::vector<Matrix<T>> vectors(cols);

    // 提取列向量
    for (int c = 0; c < cols; ++c) {
        vectors[c] = Matrix<T>(rows, 1);
        for (int r = 0; r < rows; ++r)
            vectors[c].at(r, 0) = mat.at(r, c);
    }

    // Gram-Schmidt 过程
    for (int k = 0; k < cols; ++k) {
        // 正交化
        for (int j = 0; j < k; ++j) {
            T dot = 0;
            for (int i = 0; i < rows; ++i)
                dot += vectors[j].at(i, 0) * vectors[k].at(i, 0);
            for (int i = 0; i < rows; ++i)
                vectors[k].at(i, 0) -= dot * vectors[j].at(i, 0);
        }

        // 归一化
        T norm = 0;
        for (int i = 0; i < rows; ++i)
            norm += vectors[k].at(i, 0) * vectors[k].at(i, 0);
        norm = std::sqrt(norm);

        if (norm < 1e-8)
            throw std::runtime_error("Column vector is nearly zero");

        for (int i = 0; i < rows; ++i)
            vectors[k].at(i, 0) /= norm;
    }

    // 写回矩阵
    for (int c = 0; c < cols; ++c)
        for (int r = 0; r < rows; ++r)
            mat.at(r, c) = vectors[c].at(r, 0);
}

// Explicit instantiation
template void GramSchmidtOrthogonalization<float>(Matrix<float>&);
template void GramSchmidtOrthogonalization<double>(Matrix<double>&);
