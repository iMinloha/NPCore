#include "Layers/GRU.h"
#include <cmath>
namespace CoreNNSpace {

Matrix<float> GRU::forward(Matrix<float>& input) {
    seq_len = input.row; COREPP_ASSERT(input.col == input_size, "GRU size mismatch");
    auto* res = new Matrix<float>(seq_len, hidden_size);
    if (train_mode) { h_cache.resize(seq_len); x_cache.resize(seq_len); gate_cache.resize(seq_len); }
    Matrix<float> h_prev(hidden_size, 1);
    int td = input_size + hidden_size;

    for (int t = 0; t < seq_len; ++t) {
        Matrix<float> x_t(input_size, 1), xh(td, 1);
        for (int i = 0; i < input_size; ++i) x_t.at(i,0) = input.at(t,i);
        for (int i = 0; i < input_size; ++i) xh.at(i,0) = x_t.at(i,0);
        for (int i = 0; i < hidden_size; ++i) xh.at(input_size+i,0) = h_prev.at(i,0);
        if (train_mode) x_cache[t] = x_t;

        Matrix<float> gates = (*W) * xh + (*b);
        if (train_mode) gate_cache[t] = gates;

        Matrix<float> h_t(hidden_size, 1);
        for (int i = 0; i < hidden_size; ++i) {
            float z = 1.0f / (1.0f + std::exp(-gates.at(hidden_size+i,0)));   // update
            float n_in = gates.at(2*hidden_size+i,0);
            float n = std::tanh(n_in);  // new gate (simplified: ignores r⊙h weight)
            h_t.at(i,0) = (1.0f - z) * n + z * h_prev.at(i,0);
        }
        if (train_mode) h_cache[t] = h_t;
        for (int i = 0; i < hidden_size; ++i) res->at(t,i) = h_t.at(i,0);
        h_prev = h_t;
    }
    output.push_back(res); return *res;
}

Matrix<float> GRU::backward(Matrix<float>& grad_output) {
    int td = input_size + hidden_size;
    dW = new Matrix<float>(3*hidden_size, td);
    db = new Matrix<float>(3*hidden_size, 1);
    auto* gi = new Matrix<float>(seq_len, input_size);
    Matrix<float> dh_next(hidden_size, 1);

    for (int t = seq_len-1; t >= 0; --t) {
        Matrix<float> dh(hidden_size, 1);
        for (int i = 0; i < hidden_size; ++i) {
            float g = grad_output.at(t,i) + dh_next.at(i,0);
            if (g > grad_clip) g = grad_clip; else if (g < -grad_clip) g = -grad_clip;
            dh.at(i,0) = g;
        }
        Matrix<float>& gates = gate_cache[t];
        Matrix<float>* h_prev = (t>0) ? &h_cache[t-1] : nullptr;

        Matrix<float> dg(3*hidden_size, 1);
        for (int i = 0; i < hidden_size; ++i) {
            float r = 1.0f/(1.0f+std::exp(-gates.at(i,0)));
            float z = 1.0f/(1.0f+std::exp(-gates.at(hidden_size+i,0)));
            float n = std::tanh(gates.at(2*hidden_size+i,0));
            float h_prev_val = h_prev ? h_prev->at(i,0) : 0.0f;

            dg.at(i,0) = dh.at(i,0) * (n - h_prev_val) * r * (1.0f - r); // reset (approx)
            dg.at(hidden_size+i,0) = dh.at(i,0) * (h_prev_val - n) * z * (1.0f - z); // update
            dg.at(2*hidden_size+i,0) = dh.at(i,0) * (1.0f - z) * (1.0f - n*n); // new

            dh_next.at(i,0) = dh.at(i,0) * z;
        }

        Matrix<float> xh(td, 1);
        Matrix<float>& x_t = x_cache[t];
        for (int i = 0; i < input_size; ++i) xh.at(i,0) = x_t.at(i,0);
        for (int i = 0; i < hidden_size; ++i) xh.at(input_size+i,0) = h_prev ? h_prev->at(i,0) : 0.0f;
        for (int i = 0; i < 3*hidden_size; ++i)
            for (int j = 0; j < td; ++j) dW->at(i,j) += dg.at(i,0) * xh.at(j,0);
        for (int i = 0; i < 3*hidden_size; ++i) db->at(i,0) += dg.at(i,0);

        Matrix<float> dx = W->Translate() * dg;
        for (int j = 0; j < input_size; ++j) gi->at(t,j) = dx.at(j,0);
    }
    return *gi;
}

} // namespace
