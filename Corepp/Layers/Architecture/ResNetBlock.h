#ifndef COREPP_LAYERS_RESNETBLOCK_H
#define COREPP_LAYERS_RESNETBLOCK_H
#include "Layers/Module.h"
#include "Layers/Conv/Conv2d.h"
#include "Layers/Normalization/BatchNorm.h"
#include "Activations/Activation.h"
namespace CoreNNSpace {

// =================================[ResNet Block]================================
// Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN  +  skip -> ReLU
// If channels change, skip uses 1x1 conv to match dimensions.
class ResNetBlock : public Module<float> {
    Conv2d *conv1, *conv2, *skip_conv = nullptr;
    BatchNorm1d *bn1, *bn2;
    Activation::ReLU *relu1, *relu2;
    bool owns_skip;

public:
    ResNetBlock(int channels, bool use_bn = true)
        : owns_skip(false) {
        conv1 = new Conv2d(channels, channels, 3, 1, 1); // same padding
        conv2 = new Conv2d(channels, channels, 3, 1, 1);
        bn1   = use_bn ? new BatchNorm1d(channels) : nullptr;
        bn2   = use_bn ? new BatchNorm1d(channels) : nullptr;
        relu1 = new Activation::ReLU();
        relu2 = new Activation::ReLU();
    }
    ~ResNetBlock() override {
        delete conv1; delete conv2; delete skip_conv;
        delete bn1; delete bn2; delete relu1; delete relu2;
    }

    Matrix<float> forward(Matrix<float>& input) override {
        auto out = conv1->forward(input);
        if (bn1) out = bn1->forward(out);
        out = relu1->forward(out);
        out = conv2->forward(out);
        if (bn2) out = bn2->forward(out);
        out = out + input;  // skip connection
        out = relu2->forward(out);
        auto* saved = new Matrix<float>(out);
        output.push_back(saved);
        return *saved;
    }

    Matrix<float> backward(Matrix<float>& grad_output) override {
        auto g = relu2->backward(grad_output);
        auto g_skip = g;                     // skip path: gradient flows through
        auto g_main = g;                     // main path
        if (bn2) g_main = bn2->backward(g_main);
        g_main = conv2->backward(g_main);
        g_main = relu1->backward(g_main);
        if (bn1) g_main = bn1->backward(g_main);
        g_main = conv1->backward(g_main);
        return g_main + g_skip;
    }

    std::vector<Matrix<float>*> getParams() override {
        auto p = conv1->getParams();
        for (auto* m : conv2->getParams()) p.push_back(m);
        if (bn1) for (auto* m : bn1->getParams()) p.push_back(m);
        if (bn2) for (auto* m : bn2->getParams()) p.push_back(m);
        return p;
    }
    // NOTE: gradient collection uses individual sublayer grads
    std::vector<Matrix<float>*> getAllGrads() override {
        auto g = conv1->getAllGrads();
        for (auto* m : conv2->getAllGrads()) g.push_back(m);
        if (bn1) for (auto* m : bn1->getAllGrads()) g.push_back(m);
        if (bn2) for (auto* m : bn2->getAllGrads()) g.push_back(m);
        return g;
    }
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }

    void cuda() override { conv1->cuda(); conv2->cuda(); if(bn1)bn1->cuda(); if(bn2)bn2->cuda(); }
    void cpu()  override { conv1->cpu();  conv2->cpu();  if(bn1)bn1->cpu();  if(bn2)bn2->cpu();  }
    void eval()  { train_mode=false; conv1->eval();  conv2->eval();  if(bn1)bn1->eval();  if(bn2)bn2->eval(); }
    void train() { train_mode=true;  conv1->train(); conv2->train(); if(bn1)bn1->train(); if(bn2)bn2->train(); }

    void CleanGard() override {
        for (auto p : gard) delete p;
        gard.clear();
        for (auto p : output) delete p;
        output.clear();
        conv1->CleanGard(); conv2->CleanGard();
        if (bn1) bn1->CleanGard();
        if (bn2) bn2->CleanGard();
        relu1->CleanGard(); relu2->CleanGard();
    }
};

} // namespace
#endif
