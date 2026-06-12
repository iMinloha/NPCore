#include "Layers/Activation.h"
#include <cmath>
#include <cstring>

namespace CoreNNSpace {
namespace Activation {

// =================================[ReLU]================================
Matrix<float> ReLU::forward(Matrix<float> &input) {
    auto* result = new Matrix<float>(input.row, input.col, input.channel);
    auto* derivative = new Matrix<float>(input.row, input.col, input.channel);

    int n = input.row * input.col * input.channel;

    // --- CUDA path ---
    if (cuda_relu_dispatch(result->data_ptr(), n)) {
        std::memcpy(result->data_ptr(), input.data_ptr(), n * sizeof(float));
        cuda_relu_dispatch(result->data_ptr(), n);
        for (int i = 0; i < n; ++i)
            derivative->data_ptr()[i] = (input.data_ptr()[i] > 0) ? 1.0f : 0.0f;
    } else {
        for (int c = 0; c < input.channel; ++c) {
            for (int i = 0; i < input.row; ++i) {
                for (int j = 0; j < input.col; ++j) {
                    float val = input.at(i, j, c);
                    if (val > 0) {
                        result->at(i, j, c) = val;
                        derivative->at(i, j, c) = 1.0f;
                    } else {
                        result->at(i, j, c) = 0.0f;
                        derivative->at(i, j, c) = 0.0f;
                    }
                }
            }
        }
    }

    gard.push_back(derivative);
    output.push_back(result);
    return *result;
}

Matrix<float> ReLU::backward(Matrix<float>& grad_output) {
    if (gard.empty()) return grad_output;
    return grad_output & (*gard.back());
}

Matrix<float>* ReLU::getGard() {
    return gard.empty() ? nullptr : gard.back();
}

Matrix<float>* ReLU::getOutput() {
    return output.empty() ? nullptr : output.back();
}

void ReLU::CleanGard() {
    for (auto ptr : gard) delete ptr;
    gard.clear();
    for (auto ptr : output) delete ptr;
    output.clear();
    delete weight_grad_; weight_grad_ = nullptr;
    delete bias_grad_; bias_grad_ = nullptr;
}

// =================================[Sigmoid]================================
Matrix<float> Sigmoid::forward(Matrix<float> &input) {
    auto* result = new Matrix<float>(input.row, input.col, input.channel);
    auto* derivative = new Matrix<float>(input.row, input.col, input.channel);
    int n = input.row * input.col * input.channel;

    // --- CUDA path ---
    if (cuda_sigmoid_dispatch(result->data_ptr(), n)) {
        std::memcpy(result->data_ptr(), input.data_ptr(), n * sizeof(float));
        cuda_sigmoid_dispatch(result->data_ptr(), n);
        for (int i = 0; i < n; ++i) {
            float s = result->data_ptr()[i];
            derivative->data_ptr()[i] = s * (1.0f - s);
        }
    } else {
#pragma omp parallel for collapse(3) if (n > 1000)
        for (int ch = 0; ch < input.channel; ++ch) {
            for (int i = 0; i < input.row; ++i) {
                for (int j = 0; j < input.col; ++j) {
                    float val = input.at(i, j, ch);
                    if (val > 88.0f) {
                        result->at(i, j, ch) = 1.0f;
                        derivative->at(i, j, ch) = 0.0f;
                    } else if (val < -88.0f) {
                        result->at(i, j, ch) = 0.0f;
                        derivative->at(i, j, ch) = 0.0f;
                    } else {
                        float s = 1.0f / (1.0f + std::exp(-val));
                        result->at(i, j, ch) = s;
                        derivative->at(i, j, ch) = s * (1.0f - s);
                    }
                }
            }
        }
    }

    gard.push_back(derivative);
    output.push_back(result);
    return *result;
}

Matrix<float> Sigmoid::backward(Matrix<float>& grad_output) {
    if (gard.empty()) return grad_output;
    return grad_output & (*gard.back());
}

Matrix<float>* Sigmoid::getGard() {
    return gard.empty() ? nullptr : gard.back();
}

Matrix<float>* Sigmoid::getOutput() {
    return output.empty() ? nullptr : output.back();
}

void Sigmoid::CleanGard() {
    for (auto ptr : gard) delete ptr;
    gard.clear();
    for (auto ptr : output) delete ptr;
    output.clear();
    delete weight_grad_; weight_grad_ = nullptr;
    delete bias_grad_; bias_grad_ = nullptr;
}

// =================================[SoftMax — Optimized]================================
// Uses the identity:  dL/dx_i = y_i * (dL/dy_i - sum_j dL/dy_j * y_j)
// This is O(N) per row instead of O(N^3) for the full Jacobian multiply.
// Stores softmax output in gard (not the Jacobian).

Matrix<float> SoftMax::forward(Matrix<float> &input) {
    auto* result = new Matrix<float>(input.row, input.col);

    COREPP_ASSERT(input.row > 0 && input.col > 0, "SoftMax: input shape error");

    // Softmax per row: y_i = exp(x_i - max) / sum_j exp(x_j - max)
    for (int i = 0; i < input.row; ++i) {
        float max_val = -1e30f;
        for (int j = 0; j < input.col; ++j) {
            float v = input.at(i, j);
            if (v > max_val) max_val = v;
        }

        float sum = 0.0f;
        for (int j = 0; j < input.col; ++j)
            sum += std::exp(input.at(i, j) - max_val);

        for (int j = 0; j < input.col; ++j)
            result->at(i, j) = std::exp(input.at(i, j) - max_val) / sum;
    }

    // Store softmax output for backward (NOT the full N×N Jacobian)
    gard.push_back(new Matrix<float>(*result));
    output.push_back(result);
    return *result;
}

Matrix<float> SoftMax::backward(Matrix<float>& grad_output) {
    // dL/dx_i = y_i * (g_i - sum_j g_j * y_j)   where y = softmax(x), g = dL/dy
    // This is the efficient O(N) per-row formula, avoiding the full Jacobian.

    if (gard.empty()) return grad_output;

    Matrix<float>& y = *gard.back();  // softmax output stored during forward
    auto* dx = new Matrix<float>(grad_output.row, grad_output.col);

    for (int i = 0; i < grad_output.row; ++i) {
        // Compute weighted sum: sum_j grad_output[j] * y[j]
        float weighted_sum = 0.0f;
        for (int j = 0; j < grad_output.col; ++j)
            weighted_sum += grad_output.at(i, j) * y.at(i, j);

        for (int j = 0; j < grad_output.col; ++j)
            dx->at(i, j) = y.at(i, j) * (grad_output.at(i, j) - weighted_sum);
    }

    return *dx;
}

Matrix<float>* SoftMax::getGard() {
    return gard.empty() ? nullptr : gard.back();
}

Matrix<float>* SoftMax::getOutput() {
    return output.empty() ? nullptr : output.back();
}

void SoftMax::CleanGard() {
    for (auto ptr : gard) delete ptr;
    gard.clear();
    for (auto ptr : output) delete ptr;
    output.clear();
    delete weight_grad_; weight_grad_ = nullptr;
    delete bias_grad_; bias_grad_ = nullptr;
}

} // namespace Activation
} // namespace CoreNNSpace
