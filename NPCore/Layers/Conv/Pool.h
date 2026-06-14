#ifndef NPCORE_LAYERS_POOL_H
#define NPCORE_LAYERS_POOL_H

#include "Layers/Module.h"

namespace NPCore {

// =================================[MaxPool2d]================================
class NPCORE_API MaxPool2d : public Module<float> {
    int pool_size, stride;
    int in_h, in_w, in_c;
    Matrix<float>* mask = nullptr;

public:
    MaxPool2d(int pool_size = 2, int stride = 2);
    ~MaxPool2d() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
};

// =================================[AvgPool2d]================================
class NPCORE_API AvgPool2d : public Module<float> {
    int pool_size, stride;
    int in_h, in_w, in_c;

public:
    AvgPool2d(int pool_size = 2, int stride = 2);
    ~AvgPool2d() override = default;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
};

// =================================[AdaptiveAvgPool2d]================================
class NPCORE_API AdaptiveAvgPool2d : public Module<float> {
    int output_h, output_w;
    int in_h, in_w, in_c;

public:
    AdaptiveAvgPool2d(int output_h, int output_w);
    explicit AdaptiveAvgPool2d(int output_size = 1);
    ~AdaptiveAvgPool2d() override = default;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
};

} // namespace NPCore
#endif
