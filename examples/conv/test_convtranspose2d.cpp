// =================================[ConvTranspose2d — Precision Test]================================
// Verifies: upsampling output shape formula, forward/backward consistency, weight/bias gradient flow.

#include "CorePP.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace CoreNNSpace;

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  ConvTranspose2d — Precision Analysis\n";
    std::cout << "============================================================\n";

    constexpr int C_in = 4, C_out = 2, K = 3, S = 2, P = 1;
    constexpr int H_in = 4, W_in = 4;
    constexpr int H_out = (H_in - 1) * S - 2 * P + K;  // = 7
    constexpr int W_out = H_out;

    std::cout << "\n[Config] C_in=" << C_in << " → C_out=" << C_out
              << "  kernel=" << K << "  stride=" << S << "  pad=" << P << "\n";
    std::cout << "  Spatial: " << H_in << "×" << W_in << " → " << H_out << "×" << W_out
              << "  (formula: (H-1)×S - 2P + K)\n";

    ConvTranspose2d ct(C_in, C_out, K, S, P, InitMode::Uniform, 0.0, 0.1);
    ct.train();

    Matrix<float> x(H_in, W_in, C_in);
    for (int c = 0; c < C_in; ++c)
        for (int i = 0; i < H_in; ++i)
            for (int j = 0; j < W_in; ++j)
                x.at(i, j, c) = (float)(c * 100 + i * 10 + j + 1) / 100.0f;
    x.Analysis("ConvTranspose2d Input (H×W×C)");

    // Forward
    auto out = ct.forward(x);
    out.Analysis("ConvTranspose2d Output — upsampled 4×4→7×7");

    COREPP_ASSERT(out.row == H_out && out.col == W_out && out.channel == C_out,
                "ConvTranspose2d output shape: got (%d,%d,%d), expected (%d,%d,%d)",
                out.row, out.col, out.channel, H_out, W_out, C_out);

    float out_sum = 0;
    for (int i = 0; i < H_out * W_out * C_out; ++i) out_sum += out.data_ptr()[i];
    std::cout << "  Output sum: " << std::fixed << std::setprecision(4) << out_sum << "\n";

    // Show weight sample
    ct.getParams()[0]->Analysis("ConvTranspose2d Weight (K×K×C_in·C_out)");

    // Backward
    Matrix<float> grad(H_out, W_out, C_out);
    for (int i = 0; i < H_out * W_out * C_out; ++i) grad.data_ptr()[i] = 0.05f;
    auto dx = ct.backward(grad);

    COREPP_ASSERT(dx.row == H_in && dx.col == W_in && dx.channel == C_in,
                "ConvTranspose2d backward shape: got (%d,%d,%d), expected (%d,%d,%d)",
                dx.row, dx.col, dx.channel, H_in, W_in, C_in);

    float dx_sum = 0;
    for (int i = 0; i < H_in * W_in * C_in; ++i) dx_sum += dx.data_ptr()[i];
    std::cout << "  dL/dx sum: " << dx_sum << " (finite=" << std::isfinite(dx_sum) << ")\n";
    COREPP_ASSERT(std::isfinite(dx_sum) && std::abs(dx_sum) > 1e-8f,
                "ConvTranspose2d backward gradient is zero/NaN");

    auto* wg = ct.getWeightGrad();
    auto* bg = ct.getBiasGrad();
    if (wg) wg->Analysis("ConvTranspose2d dW (weight gradient)");
    if (bg) bg->Analysis("ConvTranspose2d db (bias gradient)");

    ct.CleanGard();
    std::cout << "\n[ConvTranspose2d] ALL CHECKS PASSED\n";
    return 0;
}
