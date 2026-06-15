#ifndef NPCORE_LAYERS_RESNETBLOCK_H
#define NPCORE_LAYERS_RESNETBLOCK_H

#include "Layers/Module.h"
#include "Layers/Conv/Conv2d.h"
#include "Layers/Normalization/BatchNorm.h"
#include "Activations/Activation.h"

namespace NPCore {

// =================================[ResNet Block]================================
// Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN  +  skip -> ReLU

class NPCORE_API ResNetBlock : public Module<float> {
    Conv2d *conv1, *conv2, *skip_conv = nullptr;
    BatchNorm1d *bn1, *bn2;
    Activation::ReLU *relu1, *relu2;
    bool owns_skip;

public:
    ResNetBlock(int channels, bool use_bn = true);
    ~ResNetBlock() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override;
    std::vector<Matrix<float>*> getAllGrads() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;

    void cuda() override;
    void cpu() override;
    void eval() override;
    void train() override;
};

} // namespace NPCore
#endif
