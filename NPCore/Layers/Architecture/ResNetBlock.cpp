#include "Layers/Architecture/ResNetBlock.h"

namespace NPCore {

ResNetBlock::ResNetBlock(int channels, bool use_bn) : owns_skip(false) {
    conv1 = new Conv2d(channels, channels, 3, 1, 1);
    conv2 = new Conv2d(channels, channels, 3, 1, 1);
    bn1   = use_bn ? new BatchNorm1d(channels) : nullptr;
    bn2   = use_bn ? new BatchNorm1d(channels) : nullptr;
    relu1 = new Activation::ReLU();
    relu2 = new Activation::ReLU();
}

ResNetBlock::~ResNetBlock() {
    delete conv1; delete conv2; delete skip_conv;
    delete bn1; delete bn2; delete relu1; delete relu2;
}

Matrix<float> ResNetBlock::forward(Matrix<float>& input) {
    auto out = conv1->forward(input);
    if (bn1) out = bn1->forward(out);
    out = relu1->forward(out);
    out = conv2->forward(out);
    if (bn2) out = bn2->forward(out);
    out = out + input;
    out = relu2->forward(out);
    auto* saved = new Matrix<float>(out);
    output.push_back(saved);
    return *saved;
}

Matrix<float> ResNetBlock::backward(Matrix<float>& grad_output) {
    auto g = relu2->backward(grad_output);
    auto g_skip = g;
    auto g_main = g;
    if (bn2) g_main = bn2->backward(g_main);
    g_main = conv2->backward(g_main);
    g_main = relu1->backward(g_main);
    if (bn1) g_main = bn1->backward(g_main);
    g_main = conv1->backward(g_main);
    return g_main + g_skip;
}

std::vector<Matrix<float>*> ResNetBlock::getParams() {
    auto p = conv1->getParams();
    for (auto* m : conv2->getParams()) p.push_back(m);
    if (bn1) for (auto* m : bn1->getParams()) p.push_back(m);
    if (bn2) for (auto* m : bn2->getParams()) p.push_back(m);
    return p;
}

std::vector<Matrix<float>*> ResNetBlock::getAllGrads() {
    auto g = conv1->getAllGrads();
    for (auto* m : conv2->getAllGrads()) g.push_back(m);
    if (bn1) for (auto* m : bn1->getAllGrads()) g.push_back(m);
    if (bn2) for (auto* m : bn2->getAllGrads()) g.push_back(m);
    return g;
}

Matrix<float>* ResNetBlock::getGard() { return nullptr; }
Matrix<float>* ResNetBlock::getOutput() { return output.empty() ? nullptr : output.back(); }

void ResNetBlock::cuda() { conv1->cuda(); conv2->cuda(); if(bn1)bn1->cuda(); if(bn2)bn2->cuda(); }
void ResNetBlock::cpu()  { conv1->cpu();  conv2->cpu();  if(bn1)bn1->cpu();  if(bn2)bn2->cpu();  }
void ResNetBlock::eval()  { train_mode=false; conv1->eval();  conv2->eval();  if(bn1)bn1->eval();  if(bn2)bn2->eval(); }
void ResNetBlock::train() { train_mode=true;  conv1->train(); conv2->train(); if(bn1)bn1->train(); if(bn2)bn2->train(); }

void ResNetBlock::CleanGard() {
    for (auto p : gard) { delete p; }
    gard.clear();
    for (auto p : output) { delete p; }
    output.clear();
    conv1->CleanGard(); conv2->CleanGard();
    if (bn1) bn1->CleanGard();
    if (bn2) bn2->CleanGard();
    relu1->CleanGard(); relu2->CleanGard();
}

} // namespace NPCore
