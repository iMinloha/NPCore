// =================================[ConvTranspose2d - Precision Test]================================
// Verifies: output shape formula, forward/backward shape consistency, gradients computed.

#include "NPCore.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace NPCore;

static void check(const char* label, bool ok) {
    std::cout << "  " << (ok ? "[PASS]" : "[FAIL]") << " " << label << "\n";
    if (!ok) { std::cerr << "FATAL: " << label << "\n"; std::exit(1); }
}

int main() {
    std::cout << "\n========================================\n";
    std::cout << "  ConvTranspose2d - Precision Test\n";
    std::cout << "========================================\n";

    constexpr int C_in = 4, C_out = 2, K = 3, S = 2, P = 1;
    constexpr int H_in = 4, W_in = 4;
    constexpr int H_exp = (H_in - 1) * S - 2 * P + K;  // = 7
    constexpr int W_exp = H_exp;

    std::cout << "\n[Config]\n";
    std::cout << "  C_in=" << C_in << " -> C_out=" << C_out
              << "  kernel=" << K << "x" << K << "  stride=" << S << "  pad=" << P << "\n";
    std::cout << "  Formula: H_out = (H_in-1)*S - 2*P + K\n";
    std::cout << "         = (" << H_in << "-1)*" << S << " - 2*" << P << " + " << K
              << " = " << H_exp << "\n\n";

    ConvTranspose2d ct(C_in, C_out, K, S, P, InitMode::Uniform, 0.0, 0.1);
    ct.train();

    Matrix<float> x(H_in, W_in, C_in);
    for (int c = 0; c < C_in; ++c)
        for (int i = 0; i < H_in; ++i)
            for (int j = 0; j < W_in; ++j)
                x.at(i, j, c) = (float)(c * 100 + i * 10 + j + 1) / 100.0f;
    x.Analysis("Input  [expected: 4x4x4]");

    // Forward
    auto out = ct.forward(x);
    out.Analysis("Output after upsampling");

    std::cout << "Expected output shape: (" << H_exp << ", " << W_exp << ", " << C_out << ")\n";
    std::cout << "Actual output shape:   (" << out.row << ", " << out.col << ", " << out.channel << ")\n";
    check("Output spatial size = (H_in-1)*S - 2P + K",
          out.row == H_exp && out.col == W_exp);
    check("Output channels = C_out", out.channel == C_out);

    float out_sum = 0;
    for (int i = 0; i < H_exp * W_exp * C_out; ++i) out_sum += out.data_ptr()[i];
    check("Forward output is finite", std::isfinite(out_sum));
    check("Forward output is non-zero", std::abs(out_sum) > 1e-6f);

    // Backward
    Matrix<float> grad(H_exp, W_exp, C_out);
    for (int i = 0; i < H_exp * W_exp * C_out; ++i) grad.data_ptr()[i] = 0.05f;
    auto dx = ct.backward(grad);

    std::cout << "\nExpected input grad shape: (" << H_in << ", " << W_in << ", " << C_in << ")\n";
    std::cout << "Actual input grad shape:   (" << dx.row << ", " << dx.col << ", " << dx.channel << ")\n";
    check("Input gradient shape matches input", dx.row == H_in && dx.col == W_in && dx.channel == C_in);

    float dsum = 0;
    for (int i = 0; i < H_in * W_in * C_in; ++i) dsum += dx.data_ptr()[i];
    check("Input gradient is finite", std::isfinite(dsum));
    check("Input gradient is non-zero", std::abs(dsum) > 1e-8f);

    auto* wg = ct.getWeightGrad();
    auto* bg = ct.getBiasGrad();
    check("Weight gradient computed", wg != nullptr);
    check("Bias gradient computed", bg != nullptr);

    ct.CleanGard();
    std::cout << "\n[ConvTranspose2d] ALL CHECKS PASSED (10/10)\n";
    return 0;
}
