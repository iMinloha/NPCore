#include "Layers/Basic/Linear.h"

namespace NPCore {

Linear::Linear(int in_features, int out_features,
               InitMode mode, bool use_bias,
               double mu, double sigma)
    : use_bias(use_bias) {
    weight = new Matrix<float>(out_features, in_features);
    bias = use_bias ? new Matrix<float>(out_features, 1) : nullptr;

    InitMatrixFunc(*weight, mode, {.mu = mu, .sigma = sigma});
    if (bias) InitMatrixFunc(*bias, InitMode::Zeros);
}

Matrix<float> Linear::forward(Matrix<float> &input) {
    Matrix<float> prod = *weight * input;
    if (bias) prod = prod + *bias;
    auto* result = new Matrix<float>(prod);
    if (train_mode) gard.push_back(new Matrix<float>(input));
    output.push_back(result);
    return *result;
}

Matrix<float> Linear::backward(Matrix<float>& grad_output) {
    // 从 gard 读取保存的输入
    Matrix<float>& input_cache = *gard.back();

    // Weight gradient: dL/dW = grad_output * input^T
    weight_grad_ = new Matrix<float>(grad_output * input_cache.Translate());

    // Bias gradient: dL/db = grad_output (only if bias exists)
    if (bias) bias_grad_ = new Matrix<float>(grad_output);
    else bias_grad_ = nullptr;

    // Input gradient: dL/dx = W^T * grad_output
    return weight->Translate() * grad_output;
}

Linear::~Linear() {
    delete weight;
    delete bias;
}

std::vector<Matrix<float>*> Linear::getParams() {
    if (bias) return {weight, bias};
    return {weight};
}
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
