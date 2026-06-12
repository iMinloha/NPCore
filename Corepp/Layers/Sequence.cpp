#include "Layers/Sequence.h"

namespace CoreNNSpace {

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

} // namespace CoreNNSpace
