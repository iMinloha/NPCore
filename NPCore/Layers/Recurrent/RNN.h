#ifndef NPCORE_LAYERS_RNN_H
#define NPCORE_LAYERS_RNN_H

#include <vector>
#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace NPCore {

// =================================[RNN Recurrent Neural Network Layer]================================
// Elman RNN: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b_h)
//
// Input:  Matrix(seq_len, input_size)  - each ROW is one time step x_t (1xinput_size)
// Output: Matrix(seq_len, hidden_size) - each ROW is hidden state h_t
//
// Backward: BPTT (Backpropagation Through Time) with gradient clipping.

class NPCORE_API RNN : public Module<float> {
private:
    Matrix<float>* W_ih;
    Matrix<float>* W_hh;
    Matrix<float>* b_h;

    int input_size, hidden_size;
    std::vector<Matrix<float>> h_cache;
    std::vector<Matrix<float>> x_cache;
    int seq_len = 0;
    float grad_clip = 10.0f;

    Matrix<float>* dW_hh = nullptr;
    Matrix<float>* db_h  = nullptr;

    Matrix<float> row_to_col(const Matrix<float>& mat, int t, int dim) const;
    void col_to_row(const Matrix<float>& col, Matrix<float>& mat, int t, int dim) const;

public:
    RNN(int input_size, int hidden_size,
        InitMode mode = InitMode::XavierUniform);

    ~RNN() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;

    std::vector<Matrix<float>*> getParams() override;
    std::vector<Matrix<float>*> getAllGrads() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
    void CleanGard() override;
};

} // namespace NPCore

#endif // NPCORE_LAYERS_RNN_H
