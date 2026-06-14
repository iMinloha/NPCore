#ifndef NPCORE_LAYERS_BATCHNORM_H
#define NPCORE_LAYERS_BATCHNORM_H

#include "Layers/Normalization/NormLayerBase.h"

namespace NPCore {

// =================================[BatchNorm1d]================================
class NPCORE_API BatchNorm1d : public NormLayerBase {
    Matrix<float> *running_mean, *running_var;
    float momentum = 0.9f, eps = 1e-5f;
    bool track_running = true;

public:
    explicit BatchNorm1d(int features);
    ~BatchNorm1d() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
};

// =================================[BatchNorm2d]================================
class NPCORE_API BatchNorm2d : public NormLayerBase {
    int channels;
    Matrix<float> *running_mean, *running_var;
    float momentum = 0.9f, eps = 1e-5f;
    bool track_running = true;

public:
    explicit BatchNorm2d(int channels);
    ~BatchNorm2d() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
};

} // namespace NPCore
#endif
