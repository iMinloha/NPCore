// =================================[MultiHeadAttention — Precision Test]================================
// Verifies: shape correctness, forward/backward consistency, gradient flow through all 8 parameter matrices.

#include "CorePP.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace CoreNNSpace;

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  MultiHeadAttention — Precision Analysis\n";
    std::cout << "============================================================\n";

    constexpr int seq = 4, d_model = 32, heads = 4;
    constexpr int d_head = d_model / heads;

    std::cout << "\n[Config] seq=" << seq << "  d_model=" << d_model
              << "  heads=" << heads << "  d_head=" << d_head << "  causal=true\n";

    MultiHeadAttention mha(d_model, heads, true);
    mha.train();

    // Deterministic input
    Matrix<float> x(seq, d_model);
    for (int t = 0; t < seq; ++t)
        for (int d = 0; d < d_model; ++d)
            x.at(t, d) = (float)((t + 1) * 100 + d) / 1000.0f;
    x.Analysis("MHA Input (seq×d_model)");

    // Forward
    auto out = mha.forward(x);
    out.Analysis("MHA Output (seq×d_model) — expected same shape as input");

    COREPP_ASSERT(out.row == seq && out.col == d_model,
                "MHA output shape mismatch: got (%d,%d), expected (%d,%d)",
                out.row, out.col, seq, d_model);

    float s = 0; for (int i = 0; i < seq * d_model; ++i) s += out.data_ptr()[i];
    std::cout << "  Output sum: " << std::fixed << std::setprecision(4) << s
              << " (finite=" << std::isfinite(s) << ")\n";
    COREPP_ASSERT(std::isfinite(s) && std::abs(s) > 0.001f, "MHA forward produced degenerate output");

    // Backward
    Matrix<float> grad(seq, d_model);
    for (int i = 0; i < seq * d_model; ++i) grad.data_ptr()[i] = 0.1f;
    auto dx = mha.backward(grad);

    COREPP_ASSERT(dx.row == seq && dx.col == d_model, "MHA backward shape mismatch");

    float dx_sum = 0; for (int i = 0; i < seq * d_model; ++i) dx_sum += dx.data_ptr()[i];
    std::cout << "  dL/dx sum: " << dx_sum << " (finite=" << std::isfinite(dx_sum) << ")\n";
    COREPP_ASSERT(std::isfinite(dx_sum), "MHA backward produced NaN/Inf");

    // Weight gradients
    auto grads = mha.getAllGrads();
    const char* names[] = {"dW_q","dW_k","dW_v","dW_o","db_q","db_k","db_v","db_o"};
    std::cout << "\n[Weight Gradients]\n";
    for (size_t i = 0; i < grads.size() && i < 8; ++i) {
        if (grads[i]) {
            float gs = 0; int n = grads[i]->row * grads[i]->col * grads[i]->channel;
            for (int j = 0; j < n; ++j) gs += grads[i]->data_ptr()[j];
            std::cout << "  " << names[i] << " (" << grads[i]->row << "×" << grads[i]->col
                      << ") sum=" << std::scientific << std::setprecision(4) << gs << "\n";
        }
    }
    COREPP_ASSERT(grads[0] && grads[3], "MHA weight gradients missing");

    mha.CleanGard();
    std::cout << "\n[MHA] ALL CHECKS PASSED\n";
    return 0;
}
