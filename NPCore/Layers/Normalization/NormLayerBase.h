#ifndef NPCORE_LAYERS_NORMLAYERBASE_H
#define NPCORE_LAYERS_NORMLAYERBASE_H

#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace NPCore {

// =================================[NormLayerBase — shared scale/bias for all norms]================================
// BatchNorm1d, BatchNorm2d, LayerNorm, and GroupNorm all share:
//   gamma, beta (learnable) + dgamma, dbeta (gradients).
// The base class provides getParams(), getAllGrads(), getGard(), and CleanGard().

class NPCORE_API NormLayerBase : public Module<float> {
protected:
    Matrix<float> *gamma, *beta;
    Matrix<float> *dgamma = nullptr, *dbeta = nullptr;

public:
    NormLayerBase() = default;

    std::vector<Matrix<float>*> getParams() override;
    std::vector<Matrix<float>*> getAllGrads() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
    void CleanGard() override;
};

} // namespace NPCore

#endif
