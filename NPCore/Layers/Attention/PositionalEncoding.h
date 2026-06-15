#ifndef NPCORE_LAYERS_POSITIONALENCODING_H
#define NPCORE_LAYERS_POSITIONALENCODING_H

#include "Layers/Module.h"

namespace NPCore {

// =================================[PositionalEncoding - Transformer PE]================================
// Reference: Vaswani et al. (2017) "Attention Is All You Need"
// PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
//
// Input:  (seq_len, d_model)  — token embeddings
// Output: (seq_len, d_model)  — embeddings + positional encoding
// No learnable parameters — gradient passes through unchanged.

class NPCORE_API PositionalEncoding : public Module<float> {
    int d_model;
    float scale_;  // typically 1.0 or sqrt(d_model)

public:
    explicit PositionalEncoding(int d_model, float scale = 1.0f);
    ~PositionalEncoding() override = default;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
};

} // namespace NPCore

#endif
