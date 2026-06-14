#ifndef NPCORE_LAYERS_MULTIHEADATTENTION_H
#define NPCORE_LAYERS_MULTIHEADATTENTION_H

#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace NPCore {

// =================================[MultiHeadAttention - Transformer Core]================================
// Reference: Vaswani et al. (2017) "Attention Is All You Need"
// Input/Output: (seq_len, d_model)

class NPCORE_API MultiHeadAttention : public Module<float> {
private:
    int d_model, num_heads, d_head;
    float scale;
    Matrix<float> *W_q, *W_k, *W_v, *W_o;
    Matrix<float> *b_q, *b_k, *b_v, *b_o;
    Matrix<float> *dW_q = nullptr, *dW_k = nullptr, *dW_v = nullptr, *dW_o = nullptr;
    Matrix<float> *db_q = nullptr, *db_k = nullptr, *db_v = nullptr, *db_o = nullptr;
    Matrix<float> *Q_saved = nullptr, *K_saved = nullptr, *V_saved = nullptr;
    Matrix<float> *attn_saved = nullptr;
    Matrix<float> *concat_saved = nullptr;
    bool causal;

public:
    MultiHeadAttention(int d_model, int num_heads = 8, bool causal = false);
    ~MultiHeadAttention() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override;
    std::vector<Matrix<float>*> getAllGrads() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
};

} // namespace NPCore
#endif
