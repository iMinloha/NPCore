#ifndef NPCORE_LAYERS_BATCHNORM_H
#define NPCORE_LAYERS_BATCHNORM_H

#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace NPCore {

// =================================[BatchNorm1d]================================
class BatchNorm1d : public Module<float> {
    Matrix<float> *gamma, *beta;
    Matrix<float> *running_mean, *running_var;
    Matrix<float> *dgamma = nullptr, *dbeta = nullptr;
    float momentum = 0.9f, eps = 1e-5f;
    bool track_running = true;

public:
    explicit BatchNorm1d(int features);
    ~BatchNorm1d() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override { return {gamma, beta}; }
    std::vector<Matrix<float>*> getAllGrads() override { return {dgamma, dbeta}; }
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }
};

// =================================[BatchNorm2d]================================
// Normalizes over (H, W) per channel for 3D image tensors of shape (H, W, C).
// During training:  mu_c = mean over HxW,  sigma^2_c = var over HxW
// During eval:     uses running estimates (default momentum 0.9)
// y = gamma_c * (x - mu_c) / sqrt(sigma^2_c + eps) + beta_c

class BatchNorm2d : public Module<float> {
    int channels;
    Matrix<float> *gamma, *beta;
    Matrix<float> *running_mean, *running_var;
    Matrix<float> *dgamma = nullptr, *dbeta = nullptr;
    float momentum = 0.9f, eps = 1e-5f;
    bool track_running = true;

public:
    explicit BatchNorm2d(int channels);
    ~BatchNorm2d() override;

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
