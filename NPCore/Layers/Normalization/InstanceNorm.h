#ifndef NPCORE_LAYERS_INSTANCENORM_H
#define NPCORE_LAYERS_INSTANCENORM_H

#include "Layers/Normalization/NormLayerBase.h"

namespace NPCore {

// =================================[InstanceNorm1d]================================
// Per-sample normalization over feature dimension.
// Input: (N, C) — normalizes each sample independently

class NPCORE_API InstanceNorm1d : public NormLayerBase {
    int features;
    float eps = 1e-5f;

public:
    explicit InstanceNorm1d(int features);
    ~InstanceNorm1d() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
};

// =================================[InstanceNorm2d]================================
// Per-sample per-channel normalization over spatial dims.
// Input: (H, W, C) — normalizes each channel of each sample independently.
// Used in style transfer (Ulyanov et al., 2016).

class NPCORE_API InstanceNorm2d : public NormLayerBase {
    int channels;
    float eps = 1e-5f;

public:
    explicit InstanceNorm2d(int channels);
    ~InstanceNorm2d() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
};

} // namespace NPCore

#endif
