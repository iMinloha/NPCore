#include "Layers/LSTM.h"
#include <cmath>

namespace CoreNNSpace {

LSTM::LSTM(int input_size, int hidden_size, InitMode mode)
    : input_size(input_size), hidden_size(hidden_size) {
    int total_dim = input_size + hidden_size;
    W = new Matrix<float>(4 * hidden_size, total_dim);
    b = new Matrix<float>(4 * hidden_size, 1);
    InitMatrixFunc(*W, mode);
    InitMatrixFunc(*b, InitMode::Zeros);

    // Scale recurrent weights smaller (stability)
    for (int i = 0; i < 4 * hidden_size; ++i)
        for (int j = input_size; j < total_dim; ++j)
            W->at(i, j) *= 0.1f;

    // Forget gate bias high → remember by default
    for (int i = 0; i < hidden_size; ++i)
        b->at(i, 0) = 2.0f;
    // Input gate bias negative → closed by default (avoid noise accumulation)
    for (int i = 0; i < hidden_size; ++i)
        b->at(hidden_size + i, 0) = -2.0f;
}

Matrix<float> LSTM::forward(Matrix<float>& input) {
    seq_len = input.row;
    COREPP_ASSERT(input.col == input_size, "LSTM input size mismatch");
    int total_dim = input_size + hidden_size;

    auto* result = new Matrix<float>(seq_len, hidden_size);
    if (train_mode) {
        h_cache.resize(seq_len);
        c_cache.resize(seq_len);
        x_cache.resize(seq_len);
        gate_cache.resize(seq_len);
    }

    Matrix<float> h_prev(hidden_size, 1);  // h_0 = 0
    Matrix<float> c_prev(hidden_size, 1);  // c_0 = 0

    for (int t = 0; t < seq_len; ++t) {
        // x_t as column (input_size, 1)
        Matrix<float> x_t(input_size, 1);
        for (int i = 0; i < input_size; ++i) x_t.at(i, 0) = input.at(t, i);
        if (train_mode) x_cache[t] = x_t;

        // Concatenated input: [x_t; h_prev] (total_dim, 1)
        Matrix<float> xh(total_dim, 1);
        for (int i = 0; i < input_size; ++i) xh.at(i, 0) = x_t.at(i, 0);
        for (int i = 0; i < hidden_size; ++i) xh.at(input_size + i, 0) = h_prev.at(i, 0);

        // gates = W * xh + b  →  (4*hidden, 1)
        Matrix<float> gates = (*W) * xh + (*b);
        if (train_mode) gate_cache[t] = gates;

        // Split gates and apply activations
        Matrix<float> h_t(hidden_size, 1);
        Matrix<float> c_t(hidden_size, 1);
        for (int i = 0; i < hidden_size; ++i) {
            float f = 1.0f / (1.0f + std::exp(-gates.at(i, 0)));               // forget
            float in = 1.0f / (1.0f + std::exp(-gates.at(hidden_size + i, 0))); // input
            float o = 1.0f / (1.0f + std::exp(-gates.at(2*hidden_size + i, 0)));// output
            float g = std::tanh(gates.at(3*hidden_size + i, 0));                 // cell gate

            c_t.at(i, 0) = f * c_prev.at(i, 0) + in * g;
            h_t.at(i, 0) = o * std::tanh(c_t.at(i, 0));
        }

        if (train_mode) { h_cache[t] = h_t; c_cache[t] = c_t; }

        for (int i = 0; i < hidden_size; ++i)
            result->at(t, i) = h_t.at(i, 0);

        h_prev = h_t;
        c_prev = c_t;
    }

    output.push_back(result);
    return *result;
}

// LSTM backward (BPTT) — gradients flow through cell state AND hidden state
Matrix<float> LSTM::backward(Matrix<float>& grad_output) {
    COREPP_ASSERT(grad_output.row == seq_len, "LSTM backward seq_len mismatch");

    int total_dim = input_size + hidden_size;
    dW = new Matrix<float>(4 * hidden_size, total_dim);
    db = new Matrix<float>(4 * hidden_size, 1);
    auto* grad_input = new Matrix<float>(seq_len, input_size);

    Matrix<float> dh_next(hidden_size, 1);
    Matrix<float> dc_next(hidden_size, 1);

    for (int t = seq_len - 1; t >= 0; --t) {
        // dh_total = grad from output + future hidden
        Matrix<float> dh_total(hidden_size, 1);
        for (int i = 0; i < hidden_size; ++i) {
            float grad = grad_output.at(t, i) + dh_next.at(i, 0);
            if (grad >  grad_clip) grad =  grad_clip;
            if (grad < -grad_clip) grad = -grad_clip;
            dh_total.at(i, 0) = grad;
        }

        Matrix<float>& c_t = c_cache[t];
        Matrix<float>* c_prev = (t > 0) ? &c_cache[t-1] : nullptr;
        Matrix<float>& gates = gate_cache[t];

        // Gate derivatives
        Matrix<float> dgates(4 * hidden_size, 1);
        for (int i = 0; i < hidden_size; ++i) {
            float o = 1.0f / (1.0f + std::exp(-gates.at(2*hidden_size + i, 0)));
            float tanh_c = std::tanh(c_t.at(i, 0));

            // dc_total = dh_total * o * (1 - tanh^2(c)) + dc_next
            float dc_total = dh_total.at(i, 0) * o * (1.0f - tanh_c * tanh_c) + dc_next.at(i, 0);
            if (dc_total >  grad_clip) dc_total =  grad_clip;
            if (dc_total < -grad_clip) dc_total = -grad_clip;

            // Gate gradients
            float f  = 1.0f / (1.0f + std::exp(-gates.at(i, 0)));
            float in = 1.0f / (1.0f + std::exp(-gates.at(hidden_size + i, 0)));
            float g  = std::tanh(gates.at(3*hidden_size + i, 0));

            float c_prev_val = c_prev ? c_prev->at(i, 0) : 0.0f;

            dgates.at(i, 0) = dc_total * c_prev_val * f * (1.0f - f);                    // forget
            dgates.at(hidden_size + i, 0) = dc_total * g * in * (1.0f - in);             // input
            dgates.at(2*hidden_size + i, 0) = dh_total.at(i, 0) * tanh_c * o * (1.0f - o);// output
            dgates.at(3*hidden_size + i, 0) = dc_total * in * (1.0f - g * g);             // cell

            // Propagate to previous cell state
            dc_next.at(i, 0) = dc_total * f;
        }

        // dW += dgates * xh^T
        Matrix<float> xh(total_dim, 1);
        Matrix<float>& x_t = x_cache[t];
        for (int i = 0; i < input_size; ++i) xh.at(i, 0) = x_t.at(i, 0);
        Matrix<float>* h_prev = (t > 0) ? &h_cache[t-1] : nullptr;
        for (int i = 0; i < hidden_size; ++i) xh.at(input_size + i, 0) = h_prev ? h_prev->at(i, 0) : 0.0f;

        for (int i = 0; i < 4 * hidden_size; ++i)
            for (int j = 0; j < total_dim; ++j)
                dW->at(i, j) += dgates.at(i, 0) * xh.at(j, 0);

        // db += dgates
        for (int i = 0; i < 4 * hidden_size; ++i)
            db->at(i, 0) += dgates.at(i, 0);

        // dx_t = W_ih^T * dgates (first input_size rows of W^T)
        Matrix<float> WT_dgates = W->Translate() * dgates;
        for (int j = 0; j < input_size; ++j)
            grad_input->at(t, j) = WT_dgates.at(j, 0);

        // dh_next = W_hh^T * dgates (last hidden_size rows of W^T)
        for (int i = 0; i < hidden_size; ++i)
            dh_next.at(i, 0) = WT_dgates.at(input_size + i, 0);
    }

    return *grad_input;
}

} // namespace CoreNNSpace
