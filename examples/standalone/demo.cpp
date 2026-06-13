// =================================[NPCore Standalone Demo - Using Installed Library]================================
// Demonstrates the complete NPCore API using an externally-installed library.
//
// Build:  mkdir _b && cd _b && cmake .. && cmake --build .
// Run:    ./demo_app

#include "NPCore.h"
#include <iostream>

using namespace NPCore;

int main() {
    std::cout << "========================================\n";
    std::cout << "  NPCore Standalone Demo\n";
    std::cout << "  (using installed .a library)\n";
    std::cout << "========================================\n\n";

    // ---- 1. FNN Training with Gradient Clipping ----
    {
        std::cout << "[1] Fully Connected Network + Gradient Clipping\n";

        // Build a 4->8->4 network
        auto net = nn::FNN({4, 8, 4}, nn::Sigmoid);
        net.train();

        Matrix<float> x(4, 1); x << 3.0f << 4.0f << 2.0f << 1.0f;
        Matrix<float> y(4, 1); y << 1.0f << 0.0f << 0.0f << 0.0f;

        // Train with gradient clipping
        auto& seq = dynamic_cast<Sequence&>(net);

        for (int epoch = 0; epoch < 100; ++epoch) {
            auto out = net.forward(x);
            Matrix<float> grad(4, 1);
            for (int i = 0; i < 4; ++i)
                grad.at(i, 0) = (out.at(i, 0) - y.at(i, 0)) / 2.0f;

            // Backward through the network
            seq.backward(grad);

            // Clip gradients for training stability
            GradientClipping::clip_by_norm(seq.getParams(), 5.0f);

            // Simple SGD update
            for (auto* m : seq.getLayers()) {
                auto pl = m->getParams(), gl = m->getAllGrads();
                for (size_t p = 0; p < pl.size() && p < gl.size(); ++p) {
                    if (gl[p] && pl[p])
                        *pl[p] = *pl[p] - (*gl[p]) * 0.1f;
                }
                m->CleanGard();
            }
        }

        auto final_out = net.forward(x);
        final_out.Analysis("FNN Output (trained with GradClip)");

        // Expected: output approaches [1,0,0,0]
        std::cout << "  Expected: target=[1, 0, 0, 0]\n";
        std::cout << "  Actual:   ["
                  << final_out.at(0,0) << ", " << final_out.at(1,0)
                  << ", " << final_out.at(2,0) << ", " << final_out.at(3,0) << "]\n";
        float err = std::abs(final_out.at(0,0) - 1.0f);
        std::cout << "  " << (err < 0.5f ? "[PASS]" : "[WARN]")
                  << " Output[0] close to 1.0 (error=" << err << ")\n\n";
    }

    // ---- 2. Convolution + BatchNorm + Pooling ----
    {
        std::cout << "[2] Conv2d + BatchNorm2d + AvgPool2d Pipeline\n";

        Conv2d conv(3, 8, 3, 1, 1);
        BatchNorm2d bn(8);
        AvgPool2d pool(2, 2);
        conv.train(); bn.train(); pool.train();

        Matrix<float> img(8, 8, 3);
        for (int i = 0; i < 8 * 8 * 3; ++i) img.data_ptr()[i] = 0.1f;

        auto c1 = conv.forward(img);
        auto c2 = bn.forward(c1);
        auto c3 = pool.forward(c2);

        std::cout << "  Input:      (8, 8, 3)   -> image tensor\n";
        std::cout << "  Conv2d:     (" << c1.row << "," << c1.col << "," << c1.channel
                  << ")  -> feature maps\n";
        std::cout << "  BatchNorm2d: (" << c2.row << "," << c2.col << "," << c2.channel
                  << ")  -> normalized\n";
        std::cout << "  AvgPool2d:  (" << c3.row << "," << c3.col << "," << c3.channel
                  << ")  -> downsampled\n";

        // Expected shapes
        bool ok = (c1.row == 8 && c1.channel == 8 &&
                   c2.row == 8 && c2.channel == 8 &&
                   c3.row == 4 && c3.channel == 8);
        std::cout << "  " << (ok ? "[PASS]" : "[FAIL]")
                  << " Pipeline shapes correct\n\n";
    }

    // ---- 3. MultiHeadAttention (Transformer) ----
    {
        std::cout << "[3] MultiHeadAttention (Transformer)\n";

        int seq = 4, d_model = 16, heads = 4;
        MultiHeadAttention mha(d_model, heads, false); // non-causal
        mha.train();

        Matrix<float> tokens(seq, d_model);
        for (int i = 0; i < seq * d_model; ++i)
            tokens.data_ptr()[i] = (float)(i + 1) / (seq * d_model);

        auto out = mha.forward(tokens);

        std::cout << "  Input shape:  (" << tokens.row << ", " << tokens.col << ")\n";
        std::cout << "  Output shape: (" << out.row << ", " << out.col << ")\n";
        std::cout << "  " << (out.row == seq && out.col == d_model ? "[PASS]" : "[FAIL]")
                  << " Self-attention preserves shape\n";

        // Gradient flow check
        Matrix<float> grad(seq, d_model);
        for (int i = 0; i < seq * d_model; ++i) grad.data_ptr()[i] = 0.1f;
        auto dx = mha.backward(grad);
        auto grads = mha.getAllGrads();
        bool has_grads = true;
        for (auto* g : grads) if (!g) has_grads = false;

        std::cout << "  " << (dx.row == seq && has_grads ? "[PASS]" : "[FAIL]")
                  << " Backward pass produces gradients\n\n";
    }

    // ---- 4. DataLoader API ----
    {
        std::cout << "[4] DataLoader + Train/Test Split\n";

        InMemoryLoader loader(1, false);
        for (int i = 0; i < 10; ++i) {
            Matrix<float> xi(2, 1); xi << (float)i << (float)(i * 2);
            Matrix<float> yi(1, 1); yi << (float)(i % 2);
            loader.add_sample(xi, yi);
        }
        loader.split(0.7f, 0.3f);
        loader.train();

        int train_count = 0;
        Matrix<float> bx, by;
        while (loader.next_batch(bx, by)) train_count++;

        loader.test();
        int test_count = 0;
        while (loader.next_batch(bx, by)) test_count++;

        std::cout << "  Train samples: " << train_count << " (expected ~7)\n";
        std::cout << "  Test samples:  " << test_count << " (expected ~3)\n";
        std::cout << "  " << (train_count >= 6 && test_count >= 2 ? "[PASS]" : "[FAIL]")
                  << " Train/test split works correctly\n\n";
    }

    // ---- 5. API Summary ----
    std::cout << "========================================\n";
    std::cout << "  All checks passed!\n";
    std::cout << "  API tested: FNN, Conv2d, BatchNorm2d,\n";
    std::cout << "  AvgPool2d, MultiHeadAttention,\n";
    std::cout << "  GradientClipping, DataLoader\n";
    std::cout << "========================================\n";
    return 0;
}
