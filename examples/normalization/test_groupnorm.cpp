// =================================[GroupNorm — Precision Test]================================
// Verifies: per-group independent normalization (μ≈0, σ²≈1 within each group),
//           cross-group isolation, backward gradient flow.

#include "CorePP.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace CoreNNSpace;

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  GroupNorm — Precision Analysis\n";
    std::cout << "============================================================\n";

    constexpr int G = 2, C = 6, cpg = C / G;  // 2 groups × 3 channels
    constexpr int H = 2, W = 3;               // 2×3 spatial = 6 positions, 6×3=18 values per group
    constexpr int N_per_group = H * W * cpg;

    std::cout << "\n[Config] groups=" << G << "  channels=" << C
              << "  ch/group=" << cpg << "  spatial=" << H << "×" << W
              << "  (N per group=" << N_per_group << ")\n";

    GroupNorm gn(G, C);
    gn.train();

    // Group 0 (ch 0-2): values 0..17   Group 1 (ch 3-5): values 100..117
    Matrix<float> x(H, W, C);
    for (int c = 0; c < C; ++c) {
        int base = (c < cpg) ? 0 : 100;
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                x.at(i, j, c) = (float)(base + c * (H * W) + i * W + j);
    }
    x.Analysis("GroupNorm Input — Group 0 (ch0-2): 0..17, Group 1 (ch3-5): 100..117");

    // Forward
    auto out = gn.forward(x);
    out.Analysis("GroupNorm Output — each group independently normalized to μ≈0,σ²≈1");

    COREPP_ASSERT(out.row == H && out.col == W && out.channel == C, "GroupNorm output shape mismatch");

    // Verify per-group statistics
    std::cout << "\n[Post-normalization per-group statistics]\n";
    for (int g = 0; g < G; ++g) {
        int ch_s = g * cpg, ch_e = ch_s + cpg;
        float sum = 0, sum_sq = 0;
        for (int c = ch_s; c < ch_e; ++c)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j) {
                    float v = out.at(i, j, c);
                    sum += v; sum_sq += v * v;
                }
        float om = sum / N_per_group;
        float ov = sum_sq / N_per_group - om * om;
        std::cout << "  Group " << g << ": μ=" << std::scientific << std::setprecision(8) << om
                  << "  σ²=" << ov << "  (target: μ≈0, σ²≈1)\n";
        COREPP_ASSERT(std::abs(om) < 1e-4f, "GroupNorm mean not ~0");
        COREPP_ASSERT(std::abs(ov - 1.0f) < 2e-3f, "GroupNorm var not ~1");
    }

    // Verify cross-group isolation: group 0 mean ≠ group 1 mean before norm, both ~0 after
    std::cout << "\n  Cross-group check: both groups normalize to μ≈0 independently ✓\n";

    // Backward
    Matrix<float> g(H, W, C);
    for (int i = 0; i < H * W * C; ++i) g.data_ptr()[i] = 0.1f;
    auto dx = gn.backward(g);
    COREPP_ASSERT(dx.row == H && dx.col == W && dx.channel == C, "GroupNorm backward shape mismatch");

    float dx_sum = 0;
    for (int i = 0; i < H * W * C; ++i) dx_sum += dx.data_ptr()[i];
    std::cout << "\n  dL/dx sum: " << dx_sum << " (≈0 per-group expected)\n";

    auto grads = gn.getAllGrads();
    if (grads.size() >= 2) {
        if (grads[0]) grads[0]->Analysis("dγ (gamma gradient)");
        if (grads[1]) grads[1]->Analysis("dβ (beta gradient)");
    }

    gn.CleanGard();
    std::cout << "\n[GroupNorm] ALL CHECKS PASSED\n";
    return 0;
}
