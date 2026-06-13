#ifndef NPCORE_LAYERS_LAYERNORM_H
#define NPCORE_LAYERS_LAYERNORM_H

#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace NPCore {

// =================================[LayerNorm]================================
// Normalize across feature dimension: y = gamma * (x - mu)/sqrt(sigma^2+eps) + beta
// Unlike BatchNorm, statistics are per-sample, no running averages needed.
class LayerNorm : public Module<float> {
    Matrix<float> *gamma, *beta, *dgamma = nullptr, *dbeta = nullptr;
    int features;
    float eps = 1e-5f;

public:
    explicit LayerNorm(int features);
    ~LayerNorm() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override { return {gamma, beta}; }
    std::vector<Matrix<float>*> getAllGrads() override { return {dgamma, dbeta}; }
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }
};

} // namespace NPCore
#endif
