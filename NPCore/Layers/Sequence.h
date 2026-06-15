#ifndef NPCORE_LAYERS_SEQUENCE_H
#define NPCORE_LAYERS_SEQUENCE_H

#include <vector>
#include <utility>
#include "Layers/Module.h"

namespace NPCore {

// =================================[Sequence 序列容器]================================
// Owns a list of layers. Forwards input through each layer sequentially.
// Inherits Module<float> → can be NESTED inside another Sequence.
// Non-copyable, movable — ensures single ownership of layer pointers.

class NPCORE_API Sequence : public Module<float> {
private:
    std::vector<Module<float> *> layers;

public:
    Sequence() = default;

    explicit Sequence(std::vector<Module<float> *> layers);

    // No copying — single ownership of layer pointers
    Sequence(const Sequence&) = delete;
    Sequence& operator=(const Sequence&) = delete;

    // Move semantics
    Sequence(Sequence&& other) noexcept;
    Sequence& operator=(Sequence&& other) noexcept;

    ~Sequence() override;

    // Add a layer (transfers ownership)
    void add(Module<float> *layer);

    // Forward pass through all layers
    Matrix<float> forward(Matrix<float> &input) override;

    // Backward pass through all layers (reverse order)
    Matrix<float> backward(Matrix<float>& grad) override;

    // Get all parameter matrices from all leaf layers (flattened)
    std::vector<Matrix<float>*> getParams() override;

    // Get all gradient matrices from all leaf layers
    std::vector<Matrix<float>*> getAllGrads() override;

    // Get all leaf modules (for optimizer backward)
    std::vector<Module<float>*> modules() override;

    // Access layers
    const std::vector<Module<float>*>& getLayers() const;

    // Stored gradient / output (from last layer)
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;

    void CleanGard() override;

    // GPU / CPU / mode — propagate to all sub-layers
    void cuda() override;
    void cpu() override;
    void eval() override;
    void train() override;
};

} // namespace NPCore

#endif // NPCORE_LAYERS_SEQUENCE_H
