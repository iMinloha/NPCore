#ifndef NPCORE_LAYERS_RESNETBLOCK_H
#define NPCORE_LAYERS_RESNETBLOCK_H

#include "Layers/Module.h"
#include "Layers/Conv/Conv2d.h"
#include "Layers/Normalization/BatchNorm.h"
#include "Activations/Activation.h"

namespace NPCore {

// =================================[ResNet Block]================================
// Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN  +  skip -> ReLU

class ResNetBlock : public Module<float> {
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
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }

    void cuda() override { conv1->cuda(); conv2->cuda(); if(bn1)bn1->cuda(); if(bn2)bn2->cuda(); }
    void cpu()  override { conv1->cpu();  conv2->cpu();  if(bn1)bn1->cpu();  if(bn2)bn2->cpu();  }
    void eval()  { train_mode=false; conv1->eval();  conv2->eval();  if(bn1)bn1->eval();  if(bn2)bn2->eval(); }
    void train() { train_mode=true;  conv1->train(); conv2->train(); if(bn1)bn1->train(); if(bn2)bn2->train(); }
};

} // namespace NPCore
#endif
