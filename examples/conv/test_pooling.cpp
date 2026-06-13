// =================================[AvgPool2d + AdaptiveAvgPool2d - Precision Test]================================
// Verifies: average pooling correctness vs manual computation, gradient even-split property,
//           adaptive pooling to fixed output size.

#include "NPCore.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace NPCore;

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  AvgPool2d + AdaptiveAvgPool2d - Precision Analysis\n";
    std::cout << "============================================================\n";

    // ==================== AvgPool2d ====================
    {
        std::cout << "\n--- AvgPool2d: 4x4x2 -> 2x2x2 (pool=2, stride=2) ---\n";
        constexpr int H = 4, W = 4, C = 2;
        AvgPool2d pool(2, 2);
        pool.train();

        Matrix<float> x(H, W, C);
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j)
                    x.at(i, j, c) = (float)(c * 100 + i * 10 + j + 1);
        x.Analysis("Input 4x4x2");

        auto out = pool.forward(x);
        out.Analysis("AvgPool2d Output 2x2x2");

        // Manual verification: each output cell = avg of corresponding 2x2 input region
        float expect[2][2];
        for (int oi = 0; oi < 2; ++oi)
            for (int oj = 0; oj < 2; ++oj) {
                float sum = 0;
                for (int pi = 0; pi < 2; ++pi)
                    for (int pj = 0; pj < 2; ++pj)
                        sum += x.at(oi*2+pi, oj*2+pj, 0);
                expect[oi][oj] = sum / 4.0f;
            }

        float max_err = 0;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                max_err = std::max(max_err, std::abs(out.at(i, j, 0) - expect[i][j]));
        std::cout << "  Manual vs actual max error: " << std::scientific << max_err << "\n";
        NPCORE_ASSERT(max_err < 1e-4f, "AvgPool2d value mismatch vs manual computation");

        // Backward: gradient evenly distributed
        Matrix<float> g(2, 2, C);
        for (int i = 0; i < 2 * 2 * C; ++i) g.data_ptr()[i] = 1.0f;
        auto dx = pool.backward(g);
        dx.Analysis("AvgPool2d dL/dx - each input gets 1/4 of output grad");
        NPCORE_ASSERT(std::abs(dx.at(0,0,0) - 0.25f) < 1e-5f, "AvgPool2d backward: expected 0.25");

        pool.CleanGard();
        std::cout << "  [AvgPool2d] PASSED\n";
    }

    // ==================== AdaptiveAvgPool2d ====================
    {
        std::cout << "\n--- AdaptiveAvgPool2d: 5x3x3 -> 1x1x3 (global pooling) ---\n";
        constexpr int H = 5, W = 3, C = 3;
        AdaptiveAvgPool2d apool(1, 1);
        apool.train();

        Matrix<float> x(H, W, C);
        // Channels: 0=all-1, 1=ramp, 2=all-7
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j) {
                    if (c == 0)      x.at(i, j, c) = 1.0f;
                    else if (c == 1) x.at(i, j, c) = (float)(i * W + j + 1);
                    else             x.at(i, j, c) = 7.0f;
                }
        x.Analysis("Input 5x3x3");

        auto out = apool.forward(x);
        out.Analysis("AdaptiveAvgPool2d Output 1x1x3 (global avg per channel)");

        // Expected per-channel mean
        float expect[3] = {1.0f, (float)(1+15)/2.0f, 7.0f};  // ch1: avg of 1..15 = 8
        float max_err = 0;
        for (int c = 0; c < C; ++c)
            max_err = std::max(max_err, std::abs(out.at(0, 0, c) - expect[c]));
        std::cout << "  Expected: ch0=" << expect[0] << " ch1=" << expect[1]
                  << " ch2=" << expect[2] << "\n";
        std::cout << "  Max error: " << std::scientific << max_err << "\n";
        NPCORE_ASSERT(max_err < 1e-4f, "AdaptiveAvgPool2d value mismatch");

        // Backward with non-uniform gradient
        Matrix<float> g(1, 1, C);
        g.at(0,0,0) = 1.0f; g.at(0,0,1) = 2.0f; g.at(0,0,2) = 3.0f;
        auto dx = apool.backward(g);
        int area = H * W;
        std::cout << "  dL/dx[0,0,0]=" << dx.at(0,0,0) << " (expected " << (1.0f/area) << ")\n";
        std::cout << "  dL/dx[0,0,1]=" << dx.at(0,0,1) << " (expected " << (2.0f/area) << ")\n";
        NPCORE_ASSERT(std::abs(dx.at(0,0,0) - 1.0f/area) < 1e-5f, "AdaptiveAvgPool2d backward ch0");
        NPCORE_ASSERT(std::abs(dx.at(0,0,1) - 2.0f/area) < 1e-5f, "AdaptiveAvgPool2d backward ch1");

        apool.CleanGard();
        std::cout << "  [AdaptiveAvgPool2d] PASSED\n";
    }

    std::cout << "\n[Pooling] ALL CHECKS PASSED\n";
    return 0;
}
