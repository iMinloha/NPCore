#ifndef NPCORE_LAYERS_MULTIHEADATTENTION_H
#define NPCORE_LAYERS_MULTIHEADATTENTION_H

#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace NPCore {

// =================================[MultiHeadAttention - Transformer Core]================================
// Reference: Vaswani et al. (2017) "Attention Is All You Need"
// Input/Output: (seq_len, d_model)

class MultiHeadAttention : public Module<float> {
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

    std::vector<Matrix<float>*> getParams() override {
        return {W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o};
    }
    std::vector<Matrix<float>*> getAllGrads() override {
        return {dW_q, dW_k, dW_v, dW_o, db_q, db_k, db_v, db_o};
    }
    Matrix<float>* getGard() override { return gard.empty() ? nullptr : gard.back(); }
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }
};

} // namespace NPCore
#endif
