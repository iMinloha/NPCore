#include "Layers/Basic/Linear.h"

namespace CoreNNSpace {

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

    // 计算权重梯度: dL/dW = grad_output * input^T
    weight_grad_ = new Matrix<float>(grad_output * input_cache.Translate());

    // 计算偏置梯度: dL/db = grad_output
    bias_grad_ = new Matrix<float>(grad_output);

    // 返回输入梯度: dL/dx = W^T * grad_output
    return weight->Translate() * grad_output;
}

} // namespace CoreNNSpace
