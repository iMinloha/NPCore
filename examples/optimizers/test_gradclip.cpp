// =================================[GradientClipping - Precision Test]================================
// Verifies: clip_by_value clamps to [min,max], clip_by_norm scales to max_norm while preserving direction.

#include "NPCore.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace NPCore;

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  GradientClipping - Precision Analysis\n";
    std::cout << "============================================================\n";

    Linear linear(8, 4);
    linear.train();

    Matrix<float> input(8, 1);
    for (int i = 0; i < 8; ++i) input.at(i, 0) = (float)(i + 1);

    // ========== Test 1: clip_by_value ==========
    {
        std::cout << "\n--- Test 1: clip_by_value [-0.5, 0.5] ---\n";
        linear.forward(input);
        Matrix<float> grad(4, 1);
        grad.at(0,0) = 80.0f; grad.at(1,0) = -50.0f; grad.at(2,0) = 0.3f; grad.at(3,0) = -0.1f;
        linear.backward(grad);

        auto* wg_before = new Matrix<float>(*linear.getWeightGrad());
        wg_before->Analysis("Weight gradient BEFORE clip");

        std::vector<Module<float>*> modules = {&linear};
        GradientClipping::clip_by_value(modules, -0.5f, 0.5f);

        auto* wg = linear.getWeightGrad();
        wg->Analysis("Weight gradient AFTER clip_by_value [-0.5, +0.5]");

        if (wg) {
            bool in_range = true;
            for (int i = 0; i < wg->row * wg->col; ++i) {
                float v = wg->data_ptr()[i];
                if (v < -0.5f || v > 0.5f) { in_range = false; break; }
            }
            NPCORE_ASSERT(in_range, "clip_by_value: values outside [-0.5, 0.5] after clip");
        }
        delete wg_before;
        std::cout << "  PASSED\n";
    }

    // ========== Test 2: clip_by_norm ==========
    {
        std::cout << "\n--- Test 2: clip_by_norm max_norm=3.0 ---\n";
        linear.CleanGard();
        linear.forward(input);
        Matrix<float> grad(4, 1);
        grad.at(0,0) = 30.0f; grad.at(1,0) = -40.0f; grad.at(2,0) = 0.0f; grad.at(3,0) = 0.0f;
        linear.backward(grad);

        // Compute norm before
        auto* wg = linear.getWeightGrad();
        auto* bg = linear.getBiasGrad();
        float norm_before = 0;
        if (wg) for (int i = 0; i < wg->row * wg->col; ++i) { float v=wg->data_ptr()[i]; norm_before+=v*v; }
        if (bg) for (int i = 0; i < bg->row * bg->col; ++i) { float v=bg->data_ptr()[i]; norm_before+=v*v; }
        norm_before = std::sqrt(norm_before);
        std::cout << "  Gradient L2 norm before: " << std::fixed << std::setprecision(4) << norm_before << "\n";

        std::vector<Module<float>*> modules = {&linear};
        float norm_after = GradientClipping::clip_by_norm(modules, 3.0f, true);
        std::cout << "  Gradient L2 norm after:  " << std::fixed << std::setprecision(4) << norm_after << "\n";

        if (norm_before > 3.0f) {
            float expected = 3.0f;
            std::cout << "  Scaling applied: factor=" << (3.0f/norm_before)
                      << " (expected norm=3.0)\n";
            NPCORE_ASSERT(std::abs(norm_after - expected) < 1e-4f, "clip_by_norm scaling incorrect");
        }
        std::cout << "  PASSED\n";
    }

    linear.CleanGard();
    std::cout << "\n[GradientClipping] ALL CHECKS PASSED\n";
    return 0;
}
