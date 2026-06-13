#ifndef NPCORE_LAYERS_POOL_H
#define NPCORE_LAYERS_POOL_H

#include "Layers/Module.h"

namespace NPCore {

// =================================[MaxPool2d]================================
class MaxPool2d : public Module<float> {
    int pool_size, stride;
    int in_h, in_w, in_c;
    Matrix<float>* mask = nullptr;

public:
    MaxPool2d(int pool_size = 2, int stride = 2);
    ~MaxPool2d() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }
};

// =================================[AvgPool2d]================================
// Averages over pooling windows. Input: (H, W, C), Output: (H_out, W_out, C)

class AvgPool2d : public Module<float> {
    int pool_size, stride;
    int in_h, in_w, in_c;

public:
    AvgPool2d(int pool_size = 2, int stride = 2);
    ~AvgPool2d() override = default;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }
};

// =================================[AdaptiveAvgPool2d]================================
// Pools to a fixed output size regardless of input dimensions.
// Input: (H, W, C), Output: (output_h, output_w, C)

class AdaptiveAvgPool2d : public Module<float> {
    int output_h, output_w;
    int in_h, in_w, in_c;

public:
    AdaptiveAvgPool2d(int output_h, int output_w);
    explicit AdaptiveAvgPool2d(int output_size = 1);
    ~AdaptiveAvgPool2d() override = default;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }
};

} // namespace NPCore
#endif
