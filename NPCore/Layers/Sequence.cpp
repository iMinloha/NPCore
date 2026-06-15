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

void Sequence::add(Module<float> *layer) { layers.push_back(layer); }

// ========== Forward / Backward ==========
Matrix<float> Sequence::forward(Matrix<float> &input) {
    Matrix<float> out = input;
    for (auto &layer: layers) out = layer->forward(out);
    auto* result = new Matrix<float>(out);
    output.push_back(result);
    return *result;
}

Matrix<float> Sequence::backward(Matrix<float>& grad) {
    Matrix<float> g = grad;
    for (int i = (int)layers.size() - 1; i >= 0; --i)
        g = layers[i]->backward(g);
    return g;
}

// ========== Params / Grads / Modules ==========
std::vector<Matrix<float>*> Sequence::getParams() {
    std::vector<Matrix<float>*> all;
    for (auto* L : layers) {
        auto p = L->getParams();
        all.insert(all.end(), p.begin(), p.end());
    }
    return all;
}

std::vector<Matrix<float>*> Sequence::getAllGrads() {
    std::vector<Matrix<float>*> all;
    for (auto* L : layers) {
        auto g = L->getAllGrads();
        all.insert(all.end(), g.begin(), g.end());
    }
    return all;
}

std::vector<Module<float>*> Sequence::modules() {
    std::vector<Module<float>*> all;
    for (auto* L : layers) {
        auto sub = L->modules();
        all.insert(all.end(), sub.begin(), sub.end());
    }
    return all;
}

const std::vector<Module<float>*>& Sequence::getLayers() const { return layers; }

Matrix<float>* Sequence::getGard() {
    return layers.empty() ? nullptr : layers.back()->getGard();
}
Matrix<float>* Sequence::getOutput() {
    return output.empty() ? nullptr : output.back();
}

// ========== Cleanup ==========
void Sequence::CleanGard() {
    for (auto* L : layers) L->CleanGard();
    for (auto p : gard)   { delete p; }
    for (auto p : output) { delete p; }
    gard.clear();
    output.clear();
}

// ========== GPU / CPU / Mode ==========
void Sequence::cuda()  { for (auto* L : layers) L->cuda();  }
void Sequence::cpu()   { for (auto* L : layers) L->cpu();   }
void Sequence::eval()  { train_mode = false; for (auto* L : layers) L->eval();  }
void Sequence::train() { train_mode = true;  for (auto* L : layers) L->train(); }

} // namespace NPCore
