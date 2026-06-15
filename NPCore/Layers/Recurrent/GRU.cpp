#include "Layers/Recurrent/GRU.h"
#include <cmath>
namespace NPCore {

GRU::GRU(int input_size, int hidden_size, InitMode mode)
    : input_size(input_size), hidden_size(hidden_size) {
    int td = input_size + hidden_size;
    W = new Matrix<float>(3 * hidden_size, td);
    b = new Matrix<float>(3 * hidden_size, 1);
    InitMatrixFunc(*W, mode); InitMatrixFunc(*b, Zeros);
    for (int i = 0; i < 3 * hidden_size; ++i)
        for (int j = input_size; j < td; ++j) W->at(i, j) *= 0.1f;
    for (int i = 0; i < hidden_size; ++i) b->at(2*hidden_size + i, 0) = 2.0f;
}

GRU::~GRU() { delete W; delete b; delete dW; delete db; }

std::vector<Matrix<float>*> GRU::getParams() { return {W, b}; }
std::vector<Matrix<float>*> GRU::getAllGrads() { return {dW, db}; }
Matrix<float>* GRU::getGard() { return gard.empty() ? nullptr : gard.back(); }
Matrix<float>* GRU::getOutput() { return output.empty() ? nullptr : output.back(); }

void GRU::CleanGard() {
    for (auto p : gard) { delete p; }
    gard.clear();
    for (auto p : output) { delete p; }
    output.clear();
    delete dW; dW = nullptr;
    delete db; db = nullptr;
    h_cache.clear(); x_cache.clear(); gate_cache.clear();
}

Matrix<float> GRU::forward(Matrix<float>& input) {
    seq_len = input.row; NPCORE_ASSERT(input.col == input_size, "GRU size mismatch");
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

        // Full merged GEMM: gates = W * xh + b
        Matrix<float> gates = (*W) * xh + (*b);
        if (train_mode) gate_cache[t] = gates;

        Matrix<float> h_t(hidden_size, 1);
        for (int i = 0; i < hidden_size; ++i) {
            float r = 1.0f / (1.0f + std::exp(-gates.at(i,0)));              // reset gate σ(W_r·[x;h])
            float z = 1.0f / (1.0f + std::exp(-gates.at(hidden_size+i,0)));   // update gate σ(W_z·[x;h])

            // Proper new gate: n = tanh(W_n_x·x + W_n_h·(r⊙h_prev) + b_n)
            // gates[2H+i] computes W_n·[x;h] + b_n, but we need r⊙h_prev instead of h_prev.
            // Correction: subtract W_n_h·h_prev, add W_n_h·(r⊙h_prev)
            float n_correction = 0;
            for (int j = 0; j < hidden_size; ++j) {
                float r_j = 1.0f / (1.0f + std::exp(-gates.at(j,0)));
                float w_nh = W->at(2*hidden_size + i, input_size + j);
                n_correction += w_nh * (r_j - 1.0f) * h_prev.at(j, 0);
            }
            float n = std::tanh(gates.at(2*hidden_size + i, 0) + n_correction);

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

        // Pre-compute r[j], z[j], n[j], and n gradient contribution per dim
        std::vector<float> r(hidden_size), z(hidden_size), n_val(hidden_size);
        std::vector<float> dn_dnet(hidden_size);  // d(tanh)/d(net) = 1 - n²
        for (int i = 0; i < hidden_size; ++i) {
            r[i] = 1.0f / (1.0f + std::exp(-gates.at(i,0)));
            z[i] = 1.0f / (1.0f + std::exp(-gates.at(hidden_size+i,0)));
            // n uses the CORRECTED net input (same as forward)
            float net_n_correction = 0;
            for (int j = 0; j < hidden_size; ++j) {
                float r_j = 1.0f / (1.0f + std::exp(-gates.at(j,0)));
                float w_nh = W->at(2*hidden_size + i, input_size + j);
                net_n_correction += w_nh * (r_j - 1.0f) * (h_prev ? h_prev->at(j,0) : 0.0f);
            }
            float net_n = gates.at(2*hidden_size + i, 0) + net_n_correction;
            n_val[i] = std::tanh(net_n);
            dn_dnet[i] = 1.0f - n_val[i] * n_val[i];
        }

        Matrix<float> dg(3*hidden_size, 1);
        for (int i = 0; i < hidden_size; ++i) {
            float h_p = h_prev ? h_prev->at(i,0) : 0.0f;

            // dL/d(net_n[i]) = dL/dh[i] * dh/dn[i] * dn/d(net_n[i])
            // = dh[i] * (1 - z[i]) * (1 - n[i]²)
            float dnet_n = dh.at(i,0) * (1.0f - z[i]) * dn_dnet[i];

            // Reset gate gradient: only through n correction (r modulates h_prev in n)
            // ∂L/∂net_r[i] = Σ_k dnet_n[k] * W_nh[k,i] * h_prev[i] * r[i] * (1-r[i])
            float dr_total = 0;
            for (int k = 0; k < hidden_size; ++k) {
                float dnet_n_k = dh.at(k,0) * (1.0f - z[k]) * dn_dnet[k];
                float w_nh_ki = W->at(2*hidden_size + k, input_size + i);
                dr_total += dnet_n_k * w_nh_ki * h_p;
            }
            dg.at(i,0) = dr_total * r[i] * (1.0f - r[i]);

            // Update gate gradient: ∂L/∂net_z[i] = dh[i] * (h_prev[i] - n[i]) * z[i] * (1-z[i])
            dg.at(hidden_size+i,0) = dh.at(i,0) * (h_p - n_val[i]) * z[i] * (1.0f - z[i]);

            // New gate gradient (w.r.t. pre-correction gate value = gates[2H+i])
            dg.at(2*hidden_size+i,0) = dnet_n;

            // Propagate to previous hidden state:
            // dh_next[i] = dh[i] * z[i]  (from h_t = z*h_prev)
            //            + Σ_k dnet_n[k] * W_nh[k,i] * (r[i] - 1)  (from n correction)
            float dh_z = dh.at(i,0) * z[i];
            float dh_n = 0;
            for (int k = 0; k < hidden_size; ++k) {
                float dnet_n_k = dh.at(k,0) * (1.0f - z[k]) * dn_dnet[k];
                float w_nh_kj = W->at(2*hidden_size + k, input_size + i);
                dh_n += dnet_n_k * w_nh_kj * (r[i] - 1.0f);
            }
            dh_next.at(i,0) = dh_z + dh_n;
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
