#include "Optimizers/GradientClipping.h"
#include "Layers/Module.h"
#include <cmath>

namespace NPCore {

void GradientClipping::clip_by_norm(const std::vector<Module<float>*>& modules, float max_norm) {
    float total_norm = 0.0f;
    for (auto* m : modules) {
        auto grads = m->getAllGrads();
        for (auto* g : grads) {
            if (!g) continue;
            for (int i = 0; i < g->row * g->col * g->channel; ++i) {
                float v = g->data_ptr()[i];
                total_norm += v * v;
            }
        }
    }
    total_norm = std::sqrt(total_norm);
    if (total_norm > max_norm && total_norm > 0.0f) {
        float scale = max_norm / total_norm;
        for (auto* m : modules) {
            auto grads = m->getAllGrads();
            for (auto* g : grads) {
                if (!g) continue;
                int n = g->row * g->col * g->channel;
                for (int i = 0; i < n; ++i) g->data_ptr()[i] *= scale;
            }
        }
    }
}

float GradientClipping::clip_by_norm(const std::vector<Module<float>*>& modules,
                                      float max_norm, bool) {
    float total_norm = 0.0f;
    for (auto* m : modules) {
        auto grads = m->getAllGrads();
        for (auto* g : grads) {
            if (!g) continue;
            for (int i = 0; i < g->row * g->col * g->channel; ++i) {
                float v = g->data_ptr()[i];
                total_norm += v * v;
            }
        }
    }
    total_norm = std::sqrt(total_norm);
    if (total_norm > max_norm && total_norm > 0.0f) {
        float scale = max_norm / total_norm;
        for (auto* m : modules) {
            auto grads = m->getAllGrads();
            for (auto* g : grads) {
                if (!g) continue;
                int n = g->row * g->col * g->channel;
                for (int i = 0; i < n; ++i) g->data_ptr()[i] *= scale;
            }
        }
        return max_norm;
    }
    return total_norm;
}

void GradientClipping::clip_by_value(const std::vector<Module<float>*>& modules,
                                      float min_val, float max_val) {
    for (auto* m : modules) {
        auto grads = m->getAllGrads();
        for (auto* g : grads) {
            if (!g) continue;
            int n = g->row * g->col * g->channel;
            for (int i = 0; i < n; ++i) {
                float& v = g->data_ptr()[i];
                if (v < min_val) v = min_val;
                if (v > max_val) v = max_val;
            }
        }
    }
}

float GradientClipping::clip_by_norm(Matrix<float>& grad, float max_norm) {
    float norm = 0.0f;
    int n = grad.row * grad.col * grad.channel;
    for (int i = 0; i < n; ++i) {
        float v = grad.data_ptr()[i];
        norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm > max_norm && norm > 0.0f) {
        float scale = max_norm / norm;
        for (int i = 0; i < n; ++i) grad.data_ptr()[i] *= scale;
    }
    return norm;
}

void GradientClipping::clip_by_value(Matrix<float>& grad, float min_val, float max_val) {
    int n = grad.row * grad.col * grad.channel;
    for (int i = 0; i < n; ++i) {
        float& v = grad.data_ptr()[i];
        if (v < min_val) v = min_val;
        if (v > max_val) v = max_val;
    }
}

} // namespace NPCore
