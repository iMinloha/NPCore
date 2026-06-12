#ifndef COREPP_LAYERS_ACTIVATION_H
#define COREPP_LAYERS_ACTIVATION_H

#include <vector>
#include "Layers/Module.h"

namespace CoreNNSpace {
namespace Activation {

// =================================[ReLU 激活函数]================================
class ReLU : public Module<float> {
public:
    ReLU() = default;
    ~ReLU() override = default;  // gard, output cleaned by ~Module()

    Matrix<float> forward(Matrix<float> &input) override;

    // 反向传播: grad * derivative (derivative 在 forward 时已存储于 gard)
    Matrix<float> backward(Matrix<float>& grad_output) override;

    std::vector<Matrix<float> *> getParams() override { return {}; }
    Matrix<float> *getGard() override;
    Matrix<float> *getOutput() override;
    void CleanGard() override;
};

// =================================[Sigmoid 激活函数]================================
class Sigmoid : public Module<float> {
public:
    Sigmoid() = default;
    ~Sigmoid() override = default;  // gard, output cleaned by ~Module()

    Matrix<float> forward(Matrix<float> &input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;

    std::vector<Matrix<float> *> getParams() override { return {}; }
    Matrix<float> *getGard() override;
    Matrix<float> *getOutput() override;
    void CleanGard() override;
};

// =================================[SoftMax 激活函数]================================
class SoftMax : public Module<float> {
public:
    SoftMax() = default;
    ~SoftMax() override = default;  // gard, output cleaned by ~Module()

    Matrix<float> forward(Matrix<float> &input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;

    std::vector<Matrix<float> *> getParams() override { return {}; }
    Matrix<float> *getGard() override;
    Matrix<float> *getOutput() override;
    void CleanGard() override;
};

} // namespace Activation
} // namespace CoreNNSpace

#endif // COREPP_LAYERS_ACTIVATION_H
