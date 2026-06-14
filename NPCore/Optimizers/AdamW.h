#ifndef NPCORE_OPTIMIZERS_ADAMW_H
#define NPCORE_OPTIMIZERS_ADAMW_H

#include <vector>
#include "Layers/Module.h"

namespace NPCore {

// =================================[AdamW - Adam with Decoupled Weight Decay]================================
// Reference: Loshchilov & Hutter (2019)

class NPCORE_API AdamW {
    std::vector<Module<float>*> params;
    float lr, beta1, beta2, eps, weight_decay;
    int t = 0;

public:
    AdamW(std::vector<Module<float>*> params, float lr = 0.001f, float wd = 0.01f,
          float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    void step(Matrix<float> grad);
};

} // namespace NPCore
#endif
