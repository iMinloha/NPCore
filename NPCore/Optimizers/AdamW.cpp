#include "Optimizers/AdamW.h"
#include "Layers/Module.h"
#include <cmath>

namespace NPCore {

AdamW::AdamW(std::vector<Module<float>*> params, float lr, float wd,
             float beta1, float beta2, float eps)
    : params(params), lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(wd) {}

void AdamW::step(Matrix<float> grad) {
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
                for (int j = 0; j < param->row * param->col; ++j)
                    param->data_ptr()[j] -= lr * weight_decay * param->data_ptr()[j];
            } else {
                *param = *param - (*g) * lr;
            }
        }
        layer->CleanGard();
    }
}

} // namespace NPCore
