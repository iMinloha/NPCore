#ifndef NPCORE_LAYERS_LAYERNORM_H
#define NPCORE_LAYERS_LAYERNORM_H

#include "Layers/Normalization/NormLayerBase.h"

namespace NPCore {

// =================================[LayerNorm]================================
class NPCORE_API LayerNorm : public NormLayerBase {
    int features;
    float eps = 1e-5f;

public:
    explicit LayerNorm(int features);
    ~LayerNorm() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
};

} // namespace NPCore
#endif
