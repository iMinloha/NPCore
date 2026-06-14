#include "Layers/Basic/Linear.h"

namespace NPCore {

Linear::Linear(int in_features, int out_features, InitMode mode, double mu, double sigma) {
    weight = new Matrix<float>(out_features, in_features);
    bias = new Matrix<float>(out_features, 1);

    InitMatrixFunc(*weight, mode, {.mu = mu, .sigma = sigma});
    InitMatrixFunc(*bias, InitMode::Zeros);
}

Matrix<float> Linear::forward(Matrix<float> &input) {
    auto* result = new Matrix<float>(*weight * input + *bias);
    if (train_mode) gard.push_back(new Matrix<float>(input));
    output.push_back(result);
    return *result;
}

Matrix<float> Linear::backward(Matrix<float>& grad_output) {
    // 从 gard 读取保存的输入
    Matrix<float>& input_cache = *gard.back();

    // Weight gradient: dL/dW = grad_output * input^T
    weight_grad_ = new Matrix<float>(grad_output * input_cache.Translate());

    // Bias gradient: dL/db = grad_output
    bias_grad_ = new Matrix<float>(grad_output);

    // Input gradient: dL/dx = W^T * grad_output
    return weight->Translate() * grad_output;
}

Linear::~Linear() {
    delete weight;
    delete bias;
}

std::vector<Matrix<float>*> Linear::getParams() { return {weight, bias}; }
Matrix<float>* Linear::getGard() { return gard.empty() ? nullptr : gard.back(); }
Matrix<float>* Linear::getOutput() { return output.empty() ? nullptr : output.back(); }

void Linear::CleanGard() {
    for (auto ptr : gard) delete ptr;
    gard.clear();
    for (auto ptr : output) delete ptr;
    output.clear();
    delete weight_grad_; weight_grad_ = nullptr;
    delete bias_grad_; bias_grad_ = nullptr;
}

} // namespace NPCore
