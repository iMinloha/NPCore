#ifndef COREPP_OPTIMIZERS_GRADIENTCLIPPING_H
#define COREPP_OPTIMIZERS_GRADIENTCLIPPING_H

#include <vector>
#include <cmath>
#include <algorithm>
#include "Layers/Module.h"

namespace CoreNNSpace {

// =================================[Gradient Clipping — 梯度裁剪]================================
// Reference: Pascanu et al. (2013) "On the difficulty of training Recurrent Neural Networks"
//
// Gradient clipping prevents exploding gradients by constraining gradient magnitude.
// Essential for training stability in RNNs, LSTMs, GRUs, and Transformers.
//
// Usage:
//   auto modules = model.getModules();
//   GradientClipping::clip_by_norm(modules, 5.0f);   // clip to max norm of 5.0
//   GradientClipping::clip_by_value(modules, -1.0f, 1.0f);  // clamp to [-1, 1]

struct GradientClipping {

    // Clip gradients by global norm (L2 norm over all parameters).
    // If total_norm > max_norm, all gradients are scaled by max_norm / total_norm.
    //
    // This preserves the relative direction of the gradient while limiting its magnitude,
    // which is the recommended approach for RNN/Transformer training.
    static void clip_by_norm(std::vector<Module<float>*>& modules, float max_norm) {
        // Compute total L2 norm across all gradients
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

        // Scale if exceeds threshold
        if (total_norm > max_norm && total_norm > 0.0f) {
            float scale = max_norm / total_norm;
            for (auto* m : modules) {
                auto grads = m->getAllGrads();
                for (auto* g : grads) {
                    if (!g) continue;
                    int n = g->row * g->col * g->channel;
                    for (int i = 0; i < n; ++i)
                        g->data_ptr()[i] *= scale;
                }
            }
        }
    }

    // Convenience overload: clip gradients of a Module vector by norm, returns the norm.
    // Useful for logging:  float norm = GradientClipping::clip_by_norm(modules, 5.0f);
    //                      printf("Gradient norm: %f\n", norm);
    static float clip_by_norm(std::vector<Module<float>*>& modules,
                              float max_norm, bool /*return_norm*/) {
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
                    for (int i = 0; i < n; ++i)
                        g->data_ptr()[i] *= scale;
                }
            }
            return max_norm;  // was clipped
        }
        return total_norm;
    }

    // Clip gradients by value — element-wise clamping.
    // Each gradient element is clamped to [min_val, max_val].
    // Useful when you want to prevent any single gradient element from being too large,
    // but don't want to change the direction of the overall gradient.
    static void clip_by_value(std::vector<Module<float>*>& modules,
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

    // Clip gradients of a single matrix (for direct use before optimizer step).
    // Returns the norm before clipping.
    static float clip_by_norm(Matrix<float>& grad, float max_norm) {
        float norm = 0.0f;
        int n = grad.row * grad.col * grad.channel;
        for (int i = 0; i < n; ++i) {
            float v = grad.data_ptr()[i];
            norm += v * v;
        }
        norm = std::sqrt(norm);

        if (norm > max_norm && norm > 0.0f) {
            float scale = max_norm / norm;
            for (int i = 0; i < n; ++i)
                grad.data_ptr()[i] *= scale;
        }
        return norm;
    }

    // Clip a single matrix by value
    static void clip_by_value(Matrix<float>& grad, float min_val, float max_val) {
        int n = grad.row * grad.col * grad.channel;
        for (int i = 0; i < n; ++i) {
            float& v = grad.data_ptr()[i];
            if (v < min_val) v = min_val;
            if (v > max_val) v = max_val;
        }
    }
};

} // namespace CoreNNSpace

#endif // COREPP_OPTIMIZERS_GRADIENTCLIPPING_H
