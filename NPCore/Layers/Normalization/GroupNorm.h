#ifndef NPCORE_LAYERS_GROUPNORM_H
#define NPCORE_LAYERS_GROUPNORM_H

#include "Layers/Normalization/NormLayerBase.h"

namespace NPCore {

// =================================[GroupNorm - Group Normalization]================================
class NPCORE_API GroupNorm : public NormLayerBase {
    int num_groups, channels, channels_per_group;
    float eps = 1e-5f;

public:
    GroupNorm(int num_groups, int channels);
    ~GroupNorm() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
};

} // namespace NPCore
#endif
