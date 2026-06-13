#include "Layers/Sequence.h"

namespace NPCore {

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

// Backward pass through all layers (reverse order)
Matrix<float> Sequence::backward(Matrix<float>& grad) {
    Matrix<float> g = grad;
    for (int i = (int)layers.size() - 1; i >= 0; --i) {
        g = layers[i]->backward(g);
    }
    return g;
}

} // namespace NPCore
