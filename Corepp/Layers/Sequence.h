#ifndef COREPP_LAYERS_SEQUENCE_H
#define COREPP_LAYERS_SEQUENCE_H

#include <vector>
#include <utility>
#include "Layers/Module.h"

namespace CoreNNSpace {

// =================================[Sequence 序列容器]================================
// Owns a list of layers. Forwards input through each layer sequentially.
// Non-copyable, movable — ensures single ownership of layer pointers.

class Sequence {
private:
    std::vector<Module<float> *> layers;

public:
    Sequence() = default;

    explicit Sequence(std::vector<Module<float> *> layers) : layers(std::move(layers)) {}

    // No copying — single ownership of layer pointers
    Sequence(const Sequence&) = delete;
    Sequence& operator=(const Sequence&) = delete;

    // Move semantics
    Sequence(Sequence&& other) noexcept : layers(std::move(other.layers)) {
        other.layers.clear();
    }

    Sequence& operator=(Sequence&& other) noexcept {
        if (this != &other) {
            for (auto layer : layers) delete layer;
            layers = std::move(other.layers);
            other.layers.clear();
        }
        return *this;
    }

    ~Sequence() { for (auto &layer: layers) delete layer; }

    // Add a layer (transfers ownership)
    void add(Module<float> *layer);

    // Forward pass through all layers
    Matrix<float> forward(Matrix<float> &input);

    // Get all layers (for optimizer parameter collection)
    std::vector<Module<float> *> getParams() { return layers; }

    // GPU: move entire model
    void cuda() { for (auto* L : layers) L->cuda(); }
    void cpu()  { for (auto* L : layers) L->cpu();  }
    void eval() { for (auto* L : layers) L->eval();  }
    void train(){ for (auto* L : layers) L->train(); }
};

} // namespace CoreNNSpace

#endif // COREPP_LAYERS_SEQUENCE_H
