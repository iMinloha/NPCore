#include "Layers/Sequence.h"
#include <utility>

namespace NPCore {

Sequence::Sequence(std::vector<Module<float> *> layers) : layers(std::move(layers)) {}

Sequence::Sequence(Sequence&& other) noexcept : layers(std::move(other.layers)) {
    other.layers.clear();
}

Sequence& Sequence::operator=(Sequence&& other) noexcept {
    if (this != &other) {
        for (auto layer : layers) delete layer;
        layers = std::move(other.layers);
        other.layers.clear();
    }
    return *this;
}

Sequence::~Sequence() { for (auto &layer: layers) delete layer; }

// 尾部插入
void Sequence::add(Module<float> *layer) {
    layers.push_back(layer);
}

// 前向传播
Matrix<float> Sequence::forward(Matrix<float> &input) {
    Matrix<float> output = input;
    for (auto &layer: layers) output = layer->forward(output);
    return output;
}

// Get all layers
std::vector<Module<float> *> Sequence::getParams() { return layers; }

const std::vector<Module<float>*>& Sequence::getLayers() const { return layers; }

// Backward pass through all layers (reverse order)
Matrix<float> Sequence::backward(Matrix<float>& grad) {
    Matrix<float> g = grad;
    for (int i = (int)layers.size() - 1; i >= 0; --i) {
        g = layers[i]->backward(g);
    }
    return g;
}

// GPU / CPU / mode
void Sequence::cuda() { for (auto* L : layers) L->cuda(); }
void Sequence::cpu()  { for (auto* L : layers) L->cpu();  }
void Sequence::eval() { for (auto* L : layers) L->eval();  }
void Sequence::train(){ for (auto* L : layers) L->train(); }

} // namespace NPCore
