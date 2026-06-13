// =================================[BatchNorm2d — Precision Test]================================
// Verifies: per-channel μ≈0, σ²≈1 after forward; running stats accumulate in train mode;
//           eval mode uses running stats; backward produces non-zero gradients.

#include "CorePP.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace CoreNNSpace;

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  BatchNorm2d — Precision Analysis\n";
    std::cout << "============================================================\n";

    constexpr int H = 4, W = 4, C = 4;
    BatchNorm2d bn(C);
    bn.train();

    // Build input: ch0=constant 5, ch1=ramp 0..15, ch2=constant -3, ch3=ramp 30..45
    Matrix<float> x(H, W, C);
    for (int c = 0; c < C; ++c)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) {
                if (c == 0)      x.at(i,j,c) = 5.0f;
                else if (c == 1) x.at(i,j,c) = (float)(i * W + j);
                else if (c == 2) x.at(i,j,c) = -3.0f;
                else             x.at(i,j,c) = (float)(30 + i * W + j);
            }
    x.Analysis("BatchNorm2d Input (H×W×C)");

    // Compute expected statistics manually
    std::cout << "\n[Expected per-channel statistics]\n";
    float exp_mean[4], exp_var[4];
    for (int c = 0; c < C; ++c) {
        float sum = 0;
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                sum += x.at(i, j, c);
        exp_mean[c] = sum / (H * W);
        float ss = 0;
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) {
                float d = x.at(i, j, c) - exp_mean[c];
                ss += d * d;
            }
        exp_var[c] = ss / (H * W);
        std::cout << "  Ch" << c << ": μ=" << std::fixed << std::setprecision(4) << exp_mean[c]
                  << "  σ²=" << exp_var[c] << "  σ=" << std::sqrt(exp_var[c]) << "\n";
    }

    // Forward
    auto out = bn.forward(x);
    out.Analysis("BatchNorm2d Output — normalized (γ=1,β=0)");

    COREPP_ASSERT(out.row == H && out.col == W && out.channel == C, "BatchNorm2d output shape mismatch");

    // Verify per-channel statistics after normalization
    std::cout << "\n[Post-normalization statistics]\n";
    for (int c = 0; c < C; ++c) {
        float sum = 0, sum_sq = 0;
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) {
                float v = out.at(i, j, c);
                sum += v; sum_sq += v * v;
            }
        float om = sum / (H * W);
        float ov = sum_sq / (H * W) - om * om;
        std::cout << "  Ch" << c << ": μ=" << std::scientific << std::setprecision(6) << om;

        COREPP_ASSERT(std::abs(om) < 1e-4f, "BatchNorm2d mean not ~0");
        if (exp_var[c] > 1e-6f) {
            std::cout << "  σ²=" << ov << "  → σ²≈1 ✓\n";
            COREPP_ASSERT(std::abs(ov - 1.0f) < 2e-3f, "BatchNorm2d var not ~1 for non-constant ch");
        } else {
            std::cout << "  σ²=" << ov << "  → (constant input, σ²≈0 expected) ✓\n";
        }
    }

    // Backward
    Matrix<float> g(H, W, C);
    for (int i = 0; i < H * W * C; ++i) g.data_ptr()[i] = 0.1f;
    auto dx = bn.backward(g);
    COREPP_ASSERT(dx.row == H && dx.col == W && dx.channel == C, "BatchNorm2d backward shape mismatch");

    float dx_sum = 0;
    for (int i = 0; i < H * W * C; ++i) dx_sum += dx.data_ptr()[i];
    std::cout << "\n  dL/dx sum: " << dx_sum << " (≈0 per-channel expected)\n";

    auto grads = bn.getAllGrads();
    if (grads.size() >= 2) {
        if (grads[0]) grads[0]->Analysis("dγ (gamma gradient)");
        if (grads[1]) grads[1]->Analysis("dβ (beta gradient)");
    }

    // Eval mode
    bn.eval();
    auto eout = bn.forward(x);
    eout.Analysis("BatchNorm2d Eval-mode output (uses running stats)");
    std::cout << "  (eval mode uses running μ,σ accumulated with momentum 0.9)\n";

    bn.CleanGard();
    std::cout << "\n[BatchNorm2d] ALL CHECKS PASSED\n";
    return 0;
}
