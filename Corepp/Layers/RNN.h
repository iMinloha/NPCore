#ifndef COREPP_LAYERS_RNN_H
#define COREPP_LAYERS_RNN_H

#include <vector>
#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace CoreNNSpace {

// =================================[RNN 循环神经网络层]================================
// Elman RNN: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b_h)
//
// Input:  Matrix(seq_len, input_size)  — each ROW is one time step x_t (1×input_size)
// Output: Matrix(seq_len, hidden_size) — each ROW is hidden state h_t
//
// Backward: BPTT (Backpropagation Through Time) with gradient clipping.
// Stores all intermediate hidden states for gradient computation.
//
// Weights: W_ih(hidden×input), W_hh(hidden×hidden), b_h(hidden×1)

class RNN : public Module<float> {
private:
    Matrix<float>* W_ih;   // input-to-hidden  (hidden, input)
    Matrix<float>* W_hh;   // hidden-to-hidden (hidden, hidden)
    Matrix<float>* b_h;    // bias             (hidden, 1)

    int input_size, hidden_size;

    // Stored sequence for BPTT
    std::vector<Matrix<float>> h_cache;
    std::vector<Matrix<float>> x_cache;
    int seq_len = 0;
    float grad_clip = 10.0f;

    // Extra gradient storage for RNN (3 params)
    Matrix<float>* dW_hh = nullptr;
    Matrix<float>* db_h  = nullptr;

    // Helper: extract row t as (input_size, 1) column vector
    Matrix<float> row_to_col(const Matrix<float>& mat, int t, int dim) const;
    // Helper: copy col to row of output matrix
    void col_to_row(const Matrix<float>& col, Matrix<float>& mat, int t, int dim) const;

public:
    RNN(int input_size, int hidden_size,
        InitMode mode = InitMode::XavierUniform);

    ~RNN() override {
        delete W_ih; delete W_hh; delete b_h;
        delete dW_hh; delete db_h;
    }

    // Forward: input(seq_len, input_size) → output(seq_len, hidden_size)
    Matrix<float> forward(Matrix<float>& input) override;

    // Backward (BPTT): grad_output(seq_len, hidden_size) → grad_input(seq_len, input_size)
    Matrix<float> backward(Matrix<float>& grad_output) override;

    std::vector<Matrix<float>*> getParams() override { return {W_ih, W_hh, b_h}; }
    std::vector<Matrix<float>*> getAllGrads() override { return {weight_grad_, dW_hh, db_h}; }

    Matrix<float>* getGard() override { return gard.empty() ? nullptr : gard.back(); }
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }

    void CleanGard() override {
        for (auto ptr : gard) delete ptr;
        gard.clear();
        for (auto ptr : output) delete ptr;
        output.clear();
        delete weight_grad_; weight_grad_ = nullptr;
        delete dW_hh; dW_hh = nullptr;
        delete db_h;  db_h  = nullptr;
        h_cache.clear();
        x_cache.clear();
    }
};

} // namespace CoreNNSpace

#endif // COREPP_LAYERS_RNN_H
