#ifndef COREPP_OPTIMIZERS_ADAMW_H
#define COREPP_OPTIMIZERS_ADAMW_H
#include <cmath>
#include "Optimizers/Optimizer.h"

namespace CoreNNSpace {

// =================================[AdamW — Adam with Decoupled Weight Decay]================================
// Reference: Loshchilov & Hutter (2019) "Decoupled Weight Decay Regularization"
//
// Unlike L2 regularization (added to gradient), AdamW applies weight decay
// directly to parameters:  θ = θ - lr * (m_hat / (√v_hat + ε) + wd * θ)
//
// Usage:
//   AdamW optim(params, 0.001f, 0.01f);  // lr=0.001, weight_decay=0.01

class AdamW {
    std::vector<Module<float>*> params;
    float lr, beta1, beta2, eps, weight_decay;
    int t = 0;

public:
    AdamW(std::vector<Module<float>*> params, float lr = 0.001f, float wd = 0.01f,
          float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
        : params(params), lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(wd) {}

    void step(Matrix<float> grad) {
        t++;
        float m_corr = 1.0f / (1.0f - std::pow(beta1, t));
        float v_corr = 1.0f / (1.0f - std::pow(beta2, t));
        Matrix<float> grad_ = grad;

        for (int i = params.size() - 1; i >= 0; --i) {
            auto* layer = params[i];
            grad_ = layer->backward(grad_);
            if (layer->getParams().empty()) { layer->CleanGard(); continue; }

            auto pl = layer->getParams(), gl = layer->getAllGrads();
            if (layer->m.row == 0 && !pl.empty()) {
                layer->m = Matrix<float>(pl[0]->row, pl[0]->col);
                layer->v = Matrix<float>(pl[0]->row, pl[0]->col);
            }
            for (size_t p = 0; p < pl.size() && p < gl.size(); ++p) {
                auto* param = pl[p], *g = gl[p];
                if (!g || !param) continue;
                if (p == 0 && param->row == layer->m.row && param->col == layer->m.col) {
                    param->fused_adam_update(*g, layer->m, layer->v,
                                              beta1, beta2, m_corr, v_corr, lr, eps);
                    // AdamW: decoupled weight decay
                    for (int j = 0; j < param->row * param->col; ++j)
                        param->data_ptr()[j] -= lr * weight_decay * param->data_ptr()[j];
                } else {
                    *param = *param - (*g) * lr;
                }
            }
            layer->CleanGard();
        }
    }
};

} // namespace
#endif
