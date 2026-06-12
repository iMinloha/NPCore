#ifndef COREPP_LOSS_H
#define COREPP_LOSS_H

// =================================[CorePP/Loss — Loss Functions]================================
// Common loss functions for training neural networks.

#include <cmath>
#include <stdexcept>
#include "Core/Matrix.hpp"

namespace CoreNNSpace {

// =================================[MSE Loss]================================
template<typename T>
T mse_loss(const Matrix<T>& pred, const Matrix<T>& target) {
    if (pred.row != target.row || pred.col != target.col)
        throw std::runtime_error("MSE: shape mismatch");

    T total = 0;
    for (int i = 0; i < pred.row; ++i)
        for (int j = 0; j < pred.col; ++j) {
            T diff = pred(i, j) - target(i, j);
            total += diff * diff;
        }
    return total / static_cast<T>(pred.row * pred.col);
}

// =================================[MSE Loss Gradient]================================
template<typename T>
Matrix<T> mse_loss_grad(const Matrix<T>& pred, const Matrix<T>& target) {
    return pred - target;  // d/dx[(x-t)^2/n] ∝ 2(x-t)/n; constant factor absorbed by lr
}

// =================================[Cross-Entropy Loss (with SoftMax)]================================
// NOTE: Use this ONLY when the network output is raw logits.
// The gradient combines softmax + cross-entropy: grad = softmax(logits) - target
template<typename T>
Matrix<T> cross_entropy_loss_grad(const Matrix<T>& logits, const Matrix<T>& target) {
    if (logits.row != target.row || logits.col != target.col)
        throw std::runtime_error("CrossEntropy: shape mismatch");

    // Compute softmax from logits
    Matrix<T> softmax_out(logits.row, logits.col);
    for (int i = 0; i < logits.row; ++i) {
        T max_val = logits(i, 0);
        for (int j = 1; j < logits.col; ++j)
            if (logits(i, j) > max_val) max_val = logits(i, j);

        T sum = 0;
        for (int j = 0; j < logits.col; ++j)
            sum += std::exp(logits(i, j) - max_val);

        for (int j = 0; j < logits.col; ++j)
            softmax_out.at(i, j) = std::exp(logits(i, j) - max_val) / sum;
    }

    // Gradient of cross-entropy with softmax: y_hat - y
    return softmax_out - target;
}

} // namespace CoreNNSpace

#endif // COREPP_LOSS_H
