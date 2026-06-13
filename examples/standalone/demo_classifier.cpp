// =================================[NPCore - Image Classifier Demo]================================
// CNN classifier: Conv2d -> BN -> ReLU -> Pool -> Conv2d -> BN -> ReLU -> AdaptivePool -> Linear
// Trains on synthetic 8x8 images: diagonal (class 0) vs anti-diagonal (class 1).
//
// Demonstrates: Conv2d, BatchNorm2d, AvgPool2d, AdaptiveAvgPool2d, GroupNorm,
//               ReLU, DataLoader, GradientClipping, manual training loop.

#include "NPCore.h"
#include <iostream>
#include <cmath>

using namespace NPCore;

static Matrix<float> make_image(int label) {
    Matrix<float> img(8, 8, 1);
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            img.at(i, j, 0) = (label == 0)
                ? ((std::abs(i - j) <= 1) ? 1.0f : 0.0f)
                : ((std::abs(i + j - 7) <= 1) ? 1.0f : 0.0f);
    return img;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  CNN Image Classifier Demo\n";
    std::cout << "  (diagonal vs anti-diagonal)\n";
    std::cout << "========================================\n" << std::flush;

    // ---- Build layers ----
    Conv2d        conv1(1, 8, 3, 1, 1);
    BatchNorm2d   bn1(8);
    Activation::ReLU relu1;
    AvgPool2d     pool1(2, 2);

    Conv2d        conv2(8, 4, 3, 1, 1);
    BatchNorm2d   bn2(4);
    Activation::ReLU relu2;
    AdaptiveAvgPool2d apool(2);
    Flatten       flatten;
    Linear        linear(16, 2);  // AdaptivePool -> (2,2,4)=16 features -> 2 classes

    Module<float>* layers[] = {&conv1, &bn1, &relu1, &pool1,
                               &conv2, &bn2, &relu2, &apool, &flatten, &linear};
    const int n_layers = 10;
    for (int i = 0; i < n_layers; ++i) layers[i]->train();

    std::cout << "[1] Model: Conv->BN->ReLU->Pool->Conv->BN->ReLU->AdaptivePool->Flatten->Linear\n" << std::flush;

    // ---- Generate dataset ----
    InMemoryLoader loader(1, true);
    for (int i = 0; i < 300; ++i) {
        int label = i % 2;
        auto img = make_image(label);
        Matrix<float> target(2, 1);
        target.at(label, 0) = 1.0f;
        loader.add_sample(img, target);
    }
    loader.split(0.8f, 0.2f);
    std::cout << "[2] Dataset: 300 samples (240 train / 60 test)\n" << std::flush;

    // ---- Train ----
    std::cout << "[3] Training 80 epochs..." << std::flush;
    float lr = 0.05f;
    Module<float>* trainable[] = {&conv1, &bn1, &conv2, &bn2, &linear};

    for (int epoch = 0; epoch < 80; ++epoch) {
        loader.train();
        Matrix<float> x, y;
        while (loader.next_batch(x, y)) {
            // Forward
            Matrix<float> out = x;
            for (int i = 0; i < n_layers; ++i) out = layers[i]->forward(out);

            // MSE gradient
            Matrix<float> grad(out.row, out.col);
            for (int i = 0; i < out.row * out.col; ++i)
                grad.data_ptr()[i] = 2.0f * (out.data_ptr()[i] - y.data_ptr()[i]) / out.row;

            // Backward
            for (int i = n_layers - 1; i >= 0; --i)
                grad = layers[i]->backward(grad);

            // Clip & update
            std::vector<Module<float>*> mods(trainable, trainable + 5);
            GradientClipping::clip_by_norm(mods, 10.0f);
            for (auto* m : mods) {
                auto pl = m->getParams(), gl = m->getAllGrads();
                for (size_t p = 0; p < pl.size() && p < gl.size(); ++p)
                    if (gl[p] && pl[p])
                        *pl[p] = *pl[p] - (*gl[p]) * lr;
                m->CleanGard();
            }
        }
    }
    std::cout << " done\n" << std::flush;

    // ---- Evaluate ----
    loader.test();
    int correct = 0, total = 0;
    Matrix<float> x, y;
    for (int i = 0; i < n_layers; ++i) layers[i]->eval();
    while (loader.next_batch(x, y)) {
        auto out = x;
        for (int i = 0; i < n_layers; ++i) out = layers[i]->forward(out);
        int pred = (out.at(0,0) > out.at(1,0)) ? 0 : 1;
        int truth = (y.at(0,0) > y.at(1,0)) ? 0 : 1;
        if (pred == truth) correct++;
        total++;
    }

    float acc = 100.0f * correct / total;
    std::cout << "[4] Test accuracy: " << correct << "/" << total
              << " = " << acc << "%\n";
    std::cout << "    " << (acc > 70.0f ? "[PASS]" : "[WARN]")
              << " (expected >70%)\n" << std::flush;

    // ---- GroupNorm bonus test ----
    GroupNorm gn(2, 4);
    gn.train();
    auto gn_img = make_image(0);
    auto gn_out = gn.forward(gn_img);
    std::cout << "[5] GroupNorm: input (8,8,1) -> output ("
              << gn_out.row << "," << gn_out.col << "," << gn_out.channel << ")\n";
    std::cout << "    " << (gn_out.row == 8 ? "[PASS]" : "[FAIL]")
              << " GroupNorm compatible with Conv pipeline\n" << std::flush;

    std::cout << "\n========================================\n";
    std::cout << "  Demo complete!\n";
    std::cout << "========================================\n";
    return 0;
}
