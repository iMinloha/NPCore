#include "Layers/Recurrent/RNN.h"
#include <cmath>

namespace NPCore {

RNN::RNN(int input_size, int hidden_size, InitMode mode)
    : input_size(input_size), hidden_size(hidden_size) {
    W_ih = new Matrix<float>(hidden_size, input_size);
    W_hh = new Matrix<float>(hidden_size, hidden_size);
    b_h  = new Matrix<float>(hidden_size, 1);

    InitMatrixFunc(*W_ih, mode);
    InitMatrixFunc(*W_hh, mode);
    InitMatrixFunc(*b_h,  InitMode::Zeros);

    // Scale W_hh smaller for stability (standard RNN practice)
    for (int i = 0; i < hidden_size; ++i)
        for (int j = 0; j < hidden_size; ++j)
            W_hh->at(i, j) *= 0.1f;
}

// Extract row t from (seq_len, dim) matrix -> column vector (dim, 1)
Matrix<float> RNN::row_to_col(const Matrix<float>& mat, int t, int dim) const {
    Matrix<float> col(dim, 1);
    for (int i = 0; i < dim; ++i)
        col.at(i, 0) = mat.at(t, i);
    return col;
}

// Copy (dim, 1) column to row t of (seq_len, dim) matrix
void RNN::col_to_row(const Matrix<float>& col, Matrix<float>& mat, int t, int dim) const {
    for (int i = 0; i < dim; ++i)
        mat.at(t, i) = col.at(i, 0);
}

// =================================[Forward]================================
Matrix<float> RNN::forward(Matrix<float>& input) {
    seq_len = input.row;
    NPCORE_ASSERT(input.col == input_size, "RNN input size mismatch");

    auto* result = new Matrix<float>(seq_len, hidden_size);
    if (train_mode) {
        h_cache.resize(seq_len);
        x_cache.resize(seq_len);
    }

    Matrix<float> h_prev(hidden_size, 1);  // h_0 = zeros

    for (int t = 0; t < seq_len; ++t) {
        // x_t: column vector (input_size, 1)
        Matrix<float> x_t = row_to_col(input, t, input_size);
        if (train_mode) x_cache[t] = x_t;

        // h_t = tanh(W_ih * x_t + W_hh * h_prev + b_h)
        Matrix<float> h_t = (*W_ih) * x_t + (*W_hh) * h_prev + (*b_h);

        for (int i = 0; i < hidden_size; ++i)
            h_t.at(i, 0) = std::tanh(h_t.at(i, 0));

        if (train_mode) h_cache[t] = h_t;
        col_to_row(h_t, *result, t, hidden_size);

        h_prev = h_t;
    }

    output.push_back(result);
    return *result;
}

// =================================[Backward: BPTT]================================
Matrix<float> RNN::backward(Matrix<float>& grad_output) {
    NPCORE_ASSERT(grad_output.row == seq_len, "RNN backward: seq_len mismatch");
    NPCORE_ASSERT(grad_output.col == hidden_size, "RNN backward: hidden_size mismatch");

    // Initialize gradients
    weight_grad_ = new Matrix<float>(hidden_size, input_size);    // dW_ih
    dW_hh        = new Matrix<float>(hidden_size, hidden_size);   // dW_hh
    db_h         = new Matrix<float>(hidden_size, 1);             // db_h

    auto* grad_input = new Matrix<float>(seq_len, input_size);
    Matrix<float> dh_next(hidden_size, 1);

    for (int t = seq_len - 1; t >= 0; --t) {
        Matrix<float> dh_total(hidden_size, 1);
        for (int i = 0; i < hidden_size; ++i) {
            float grad = grad_output.at(t, i) + dh_next.at(i, 0);
            if (grad >  grad_clip) grad =  grad_clip;
            if (grad < -grad_clip) grad = -grad_clip;
            dh_total.at(i, 0) = grad;
        }

        Matrix<float>& h_t = h_cache[t];
        Matrix<float> dtanh(hidden_size, 1);
        for (int i = 0; i < hidden_size; ++i)
            dtanh.at(i, 0) = dh_total.at(i, 0) * (1.0f - h_t.at(i, 0) * h_t.at(i, 0));

        // dW_ih += dtanh * x_t^T
        Matrix<float>& x_t = x_cache[t];
        for (int i = 0; i < hidden_size; ++i)
            for (int j = 0; j < input_size; ++j)
                weight_grad_->at(i, j) += dtanh.at(i, 0) * x_t.at(j, 0);

        // dW_hh += dtanh * h_{t-1}^T
        Matrix<float>* h_prev = (t > 0) ? &h_cache[t-1] : nullptr;
        for (int i = 0; i < hidden_size; ++i)
            for (int j = 0; j < hidden_size; ++j)
                dW_hh->at(i, j) += dtanh.at(i, 0) * (h_prev ? h_prev->at(j, 0) : 0.0f);

        // db_h += dtanh
        for (int i = 0; i < hidden_size; ++i)
            db_h->at(i, 0) += dtanh.at(i, 0);

        // dx_t = W_ih^T * dtanh
        Matrix<float> dx_t = W_ih->Translate() * dtanh;
        for (int j = 0; j < input_size; ++j)
            grad_input->at(t, j) = dx_t.at(j, 0);

        // dh_next = W_hh^T * dtanh
        dh_next = W_hh->Translate() * dtanh;
    }

    return *grad_input;
}

} // namespace NPCore
