#include "Layers/Normalization/NormLayerBase.h"

namespace NPCore {

std::vector<Matrix<float>*> NormLayerBase::getParams() { return {gamma, beta}; }
std::vector<Matrix<float>*> NormLayerBase::getAllGrads() { return {dgamma, dbeta}; }
Matrix<float>* NormLayerBase::getGard() { return nullptr; }
Matrix<float>* NormLayerBase::getOutput() { return output.empty() ? nullptr : output.back(); }

void NormLayerBase::CleanGard() {
    for (auto p : gard) { delete p; }
    gard.clear();
    for (auto p : output) { delete p; }
    output.clear();
    delete dgamma; dgamma = nullptr;
    delete dbeta;  dbeta  = nullptr;
}

} // namespace NPCore
