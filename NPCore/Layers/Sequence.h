#ifndef NPCORE_LAYERS_SEQUENCE_H
#define NPCORE_LAYERS_SEQUENCE_H

#include <vector>
#include <utility>
#include "Layers/Module.h"

namespace NPCore {

// =================================[Sequence 序列容器]================================
// Owns a list of layers. Forwards input through each layer sequentially.
// Non-copyable, movable - ensures single ownership of layer pointers.

class Sequence {
private:
    std::vector<Module<float> *> layers;

public:
    Sequence() = default;

    explicit Sequence(std::vector<Module<float> *> layers);

    // No copying - single ownership of layer pointers
    Sequence(const Sequence&) = delete;
    Sequence& operator=(const Sequence&) = delete;

    // Move semantics
    Sequence(Sequence&& other) noexcept;
    Sequence& operator=(Sequence&& other) noexcept;

    ~Sequence();

    // Add a layer (transfers ownership)
    void add(Module<float> *layer);

    // Forward pass through all layers
    Matrix<float> forward(Matrix<float> &input);

    // Get all layers (for optimizer parameter collection)
    std::vector<Module<float> *> getParams();

    // Access layers for gradient clipping, custom training loops
    const std::vector<Module<float>*>& getLayers() const;

    // Backward pass through all layers (reverse order)
    Matrix<float> backward(Matrix<float>& grad);

    // GPU: move entire model
    void cuda();
    void cpu();
    void eval();
    void train();
};

} // namespace NPCore

#endif // NPCORE_LAYERS_SEQUENCE_H
