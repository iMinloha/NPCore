#ifndef NPCORE_LAYERS_TRANSFORMERENCODER_H
#define NPCORE_LAYERS_TRANSFORMERENCODER_H

#include "Layers/Module.h"
#include "Layers/Attention/MultiHeadAttention.h"
#include "Layers/Normalization/LayerNorm.h"
#include "Layers/Basic/Linear.h"
#include "Activations/Activation.h"

namespace NPCore {

// =================================[TransformerEncoderLayer]================================
// One Transformer encoder block:
//   x = x + MHA(LayerNorm(x))         — self-attention + residual
//   x = x + FFN(LayerNorm(x))         — feed-forward + residual
//
// FFN: Linear(d_model, dim_feedforward) → GELU → Linear(dim_feedforward, d_model)

class NPCORE_API TransformerEncoderLayer : public Module<float> {
    MultiHeadAttention* mha;
    LayerNorm *norm1, *norm2;
    Linear *fc1, *fc2;
    Activation::GELU* gelu;
    int d_model;

    Matrix<float>* mha_out = nullptr;
    Matrix<float>* ffn_out = nullptr;

public:
    TransformerEncoderLayer(int d_model, int nhead = 8, int dim_ff = 2048);
    ~TransformerEncoderLayer() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override;
    std::vector<Matrix<float>*> getAllGrads() override;
    std::vector<Module<float>*> modules() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;

    void cuda() override;
    void cpu() override;
    void eval() override;
    void train() override;
};

// =================================[TransformerEncoder]================================
// Stack of TransformerEncoderLayer blocks.
// Convenience wrapper: creates num_layers identical blocks.

class NPCORE_API TransformerEncoder : public Module<float> {
    std::vector<TransformerEncoderLayer*> layers;

public:
    TransformerEncoder(int d_model, int nhead = 8, int dim_ff = 2048, int num_layers = 6);
    ~TransformerEncoder() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override;
    std::vector<Matrix<float>*> getAllGrads() override;
    std::vector<Module<float>*> modules() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;

    void cuda() override;
    void cpu() override;
    void eval() override;
    void train() override;
};

} // namespace NPCore

#endif
