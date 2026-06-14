#ifndef NPCORE_LAYERS_GROUPNORM_H
#define NPCORE_LAYERS_GROUPNORM_H

#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace NPCore {

// =================================[GroupNorm - Group Normalization]================================
// Reference: Wu & He (2018) "Group Normalization"
// Divides channels into groups and normalizes within each group.
// Input: (H, W, C), Output: (H, W, C)
// num_groups must divide channels (typically 32 or channels/2).

class GroupNorm : public Module<float> {
    int num_groups, channels, channels_per_group;
    Matrix<float> *gamma, *beta;
    Matrix<float> *dgamma = nullptr, *dbeta = nullptr;
    float eps = 1e-5f;

public:
    GroupNorm(int num_groups, int channels);
    ~GroupNorm() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override;
    std::vector<Matrix<float>*> getAllGrads() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
};

} // namespace NPCore
#endif
