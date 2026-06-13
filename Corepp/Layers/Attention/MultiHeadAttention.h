#ifndef COREPP_LAYERS_MULTIHEADATTENTION_H
#define COREPP_LAYERS_MULTIHEADATTENTION_H

#include <vector>
#include <cmath>
#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace CoreNNSpace {

// =================================[MultiHeadAttention — Transformer Core]================================
// Reference: Vaswani et al. (2017) "Attention Is All You Need"
//
// y = Concat(head_0, ..., head_{h-1}) * W_o
// head_i = Attention(Q_i, K_i, V_i) = softmax(Q_i K_i^T / √d_head + mask) * V_i
//
// Input shape:  (seq_len, d_model) — 2D matrix, rows=tokens, cols=features
// Output shape: (seq_len, d_model) — same shape as input
//
// When causal=true, a lower-triangular mask is applied so position t only attends
// to positions ≤ t (used in decoder / autoregressive generation).

class MultiHeadAttention : public Module<float> {
private:
    int d_model, num_heads, d_head;
    float scale;

    // Projection weights: (d_model, d_model)
    Matrix<float> *W_q, *W_k, *W_v, *W_o;
    Matrix<float> *b_q, *b_k, *b_v, *b_o;

    // Accumulated gradients (set by backward, read by optimizer)
    Matrix<float> *dW_q = nullptr, *dW_k = nullptr, *dW_v = nullptr, *dW_o = nullptr;
    Matrix<float> *db_q = nullptr, *db_k = nullptr, *db_v = nullptr, *db_o = nullptr;

    // Saved intermediates for backward
    Matrix<float> *Q_saved = nullptr, *K_saved = nullptr, *V_saved = nullptr;
    Matrix<float> *attn_saved = nullptr;  // attention weights after softmax
    Matrix<float> *concat_saved = nullptr; // concatenated heads before W_o

    bool causal;

public:
    MultiHeadAttention(int d_model, int num_heads = 8, bool causal = false)
        : d_model(d_model), num_heads(num_heads), d_head(d_model / num_heads),
          scale(1.0f / std::sqrt(static_cast<float>(d_head))), causal(causal)
    {
        COREPP_ASSERT(d_model % num_heads == 0,
                    "d_model (%d) must be divisible by num_heads (%d)", d_model, num_heads);

        W_q = new Matrix<float>(d_model, d_model);
        W_k = new Matrix<float>(d_model, d_model);
        W_v = new Matrix<float>(d_model, d_model);
        W_o = new Matrix<float>(d_model, d_model);

        b_q = new Matrix<float>(d_model, 1);
        b_k = new Matrix<float>(d_model, 1);
        b_v = new Matrix<float>(d_model, 1);
        b_o = new Matrix<float>(d_model, 1);

        // Xavier uniform init for weights, zeros for biases
        InitMatrixFunc(*W_q, InitMode::XavierUniform);
        InitMatrixFunc(*W_k, InitMode::XavierUniform);
        InitMatrixFunc(*W_v, InitMode::XavierUniform);
        InitMatrixFunc(*W_o, InitMode::XavierUniform);
        InitMatrixFunc(*b_q, InitMode::Zeros);
        InitMatrixFunc(*b_k, InitMode::Zeros);
        InitMatrixFunc(*b_v, InitMode::Zeros);
        InitMatrixFunc(*b_o, InitMode::Zeros);
    }

    ~MultiHeadAttention() override {
        delete W_q; delete W_k; delete W_v; delete W_o;
        delete b_q; delete b_k; delete b_v; delete b_o;
        delete dW_q; delete dW_k; delete dW_v; delete dW_o;
        delete db_q; delete db_k; delete db_v; delete db_o;
        delete Q_saved; delete K_saved; delete V_saved;
        delete attn_saved; delete concat_saved;
    }

    // ===============================[Forward]===============================
    Matrix<float> forward(Matrix<float>& input) override {
        const int seq_len = input.row;
        const int D = d_model;
        const int H = num_heads;
        const int Dh = d_head;

        // 1. Linear projections: Q, K, V  → each (d_model, seq_len)
        //   input is (seq_len, d_model), transpose to (d_model, seq_len) for matmul
        Matrix<float> input_T = input.Translate();  // (d_model, seq_len)

        Matrix<float> Q_raw = (*W_q) * input_T;     // (d_model, d_model)*(d_model, seq_len) = (d_model, seq_len)
        Matrix<float> K_raw = (*W_k) * input_T;
        Matrix<float> V_raw = (*W_v) * input_T;

        // Add bias: broadcast (d_model, 1) → (d_model, seq_len)
        for (int d = 0; d < D; ++d) {
            float bq = b_q->at(d, 0), bk = b_k->at(d, 0), bv = b_v->at(d, 0);
            for (int t = 0; t < seq_len; ++t) {
                Q_raw.at(d, t) += bq;
                K_raw.at(d, t) += bk;
                V_raw.at(d, t) += bv;
            }
        }

        // 2. Reshape to multi-head: (d_model, seq_len) → (num_heads, d_head, seq_len)
        //    Store as (d_head, seq_len, num_heads) using channel dimension
        Matrix<float> Q_mh(Dh, seq_len, H);
        Matrix<float> K_mh(Dh, seq_len, H);
        Matrix<float> V_mh(Dh, seq_len, H);

        for (int h = 0; h < H; ++h) {
            for (int d = 0; d < Dh; ++d) {
                int src_row = h * Dh + d;
                for (int t = 0; t < seq_len; ++t) {
                    Q_mh.at(d, t, h) = Q_raw.at(src_row, t);
                    K_mh.at(d, t, h) = K_raw.at(src_row, t);
                    V_mh.at(d, t, h) = V_raw.at(src_row, t);
                }
            }
        }

        // 3. Scaled dot-product attention per head → output_mh: (d_head, seq_len, num_heads)
        Matrix<float> out_mh(Dh, seq_len, H);
        Matrix<float> attn_weights(seq_len, seq_len, H);  // saved for backward

        for (int h = 0; h < H; ++h) {
            // Q_h, K_h, V_h: (d_head, seq_len)
            // scores = K_h^T * Q_h: (seq_len, d_head)*(d_head, seq_len) = (seq_len, seq_len)
            Matrix<float> K_h_T = K_mh.getChannel(h).Translate();  // (seq_len, d_head)
            Matrix<float> Q_h = Q_mh.getChannel(h);                 // (d_head, seq_len)

            Matrix<float> scores = K_h_T * Q_h;  // (seq_len, seq_len)

            // Scale
            for (int i = 0; i < seq_len; ++i)
                for (int j = 0; j < seq_len; ++j)
                    scores.at(i, j) *= scale;

            // Causal mask: set upper triangle to -inf
            if (causal) {
                for (int i = 0; i < seq_len; ++i)
                    for (int j = i + 1; j < seq_len; ++j)
                        scores.at(i, j) = -1e30f;
            }

            // Softmax row-wise (numerically stable)
            for (int i = 0; i < seq_len; ++i) {
                float max_val = -1e30f;
                for (int j = 0; j < seq_len; ++j)
                    if (scores.at(i, j) > max_val) max_val = scores.at(i, j);

                float sum_exp = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    float v = std::exp(scores.at(i, j) - max_val);
                    scores.at(i, j) = v;
                    sum_exp += v;
                }
                for (int j = 0; j < seq_len; ++j)
                    scores.at(i, j) /= sum_exp;
            }

            // Save attention weights
            for (int i = 0; i < seq_len; ++i)
                for (int j = 0; j < seq_len; ++j)
                    attn_weights.at(i, j, h) = scores.at(i, j);

            // output_h = V_h * attn_h: (d_head, seq_len)*(seq_len, seq_len) = (d_head, seq_len)
            Matrix<float> V_h = V_mh.getChannel(h);
            Matrix<float> out_h = V_h * scores;
            for (int d = 0; d < Dh; ++d)
                for (int t = 0; t < seq_len; ++t)
                    out_mh.at(d, t, h) = out_h.at(d, t);
        }

        // 4. Concat heads: (d_head, seq_len, num_heads) → (d_model, seq_len)
        Matrix<float> concat(D, seq_len);
        for (int h = 0; h < H; ++h) {
            for (int d = 0; d < Dh; ++d) {
                int dst_row = h * Dh + d;
                for (int t = 0; t < seq_len; ++t)
                    concat.at(dst_row, t) = out_mh.at(d, t, h);
            }
        }

        // 5. Output projection: (d_model, seq_len) → (d_model, seq_len)
        Matrix<float> result_raw = (*W_o) * concat;  // (d_model, seq_len)
        for (int d = 0; d < D; ++d)
            for (int t = 0; t < seq_len; ++t)
                result_raw.at(d, t) += b_o->at(d, 0);

        // Transpose back to (seq_len, d_model)
        auto* out = new Matrix<float>(seq_len, D);
        for (int t = 0; t < seq_len; ++t)
            for (int d = 0; d < D; ++d)
                out->at(t, d) = result_raw.at(d, t);

        // Save intermediates for backward
        if (train_mode) {
            gard.push_back(new Matrix<float>(input));   // original input

            delete Q_saved;
            Q_saved = new Matrix<float>(Q_raw);

            delete K_saved;
            K_saved = new Matrix<float>(K_raw);

            delete V_saved;
            V_saved = new Matrix<float>(V_raw);

            delete attn_saved;
            attn_saved = new Matrix<float>(attn_weights);

            delete concat_saved;
            concat_saved = new Matrix<float>(concat);
        }

        output.push_back(out);
        return *out;
    }

    // ===============================[Backward]===============================
    Matrix<float> backward(Matrix<float>& grad_output) override {
        const int seq_len = grad_output.row;
        const int D = d_model;
        const int H = num_heads;
        const int Dh = d_head;

        Matrix<float>& input = *gard.back();
        Matrix<float>& Q_raw = *Q_saved;
        Matrix<float>& K_raw = *K_saved;
        Matrix<float>& V_raw = *V_saved;
        Matrix<float>& attn_w = *attn_saved;   // (seq_len, seq_len, num_heads)
        Matrix<float>& concat = *concat_saved;  // (d_model, seq_len)

        // --- 1. Gradient through output projection (W_o, b_o) ---
        // grad_output is (seq_len, d_model); result_raw was (d_model, seq_len)
        // d_result = grad_output^T: (d_model, seq_len)
        Matrix<float> d_result(D, seq_len);
        for (int t = 0; t < seq_len; ++t)
            for (int d = 0; d < D; ++d)
                d_result.at(d, t) = grad_output.at(t, d);

        // dW_o = d_result * concat^T: (d_model, seq_len)*(seq_len, d_model) = (d_model, d_model)
        dW_o = new Matrix<float>(D, D);
        Matrix<float> concat_T = concat.Translate();  // (seq_len, d_model)
        Matrix<float> dW_o_tmp = d_result * concat_T;
        for (int i = 0; i < D; ++i)
            for (int j = 0; j < D; ++j)
                dW_o->at(i, j) = dW_o_tmp.at(i, j);

        // db_o = row-sum of d_result: (d_model, 1)
        db_o = new Matrix<float>(D, 1);
        for (int d = 0; d < D; ++d) {
            float s = 0;
            for (int t = 0; t < seq_len; ++t) s += d_result.at(d, t);
            db_o->at(d, 0) = s;
        }

        // d_concat = W_o^T * d_result: (d_model, d_model)^T * (d_model, seq_len) = (d_model, seq_len)
        Matrix<float> W_o_T = W_o->Translate();
        Matrix<float> d_concat = W_o_T * d_result;  // (d_model, seq_len)

        // --- 2. Gradient through attention heads ---
        // Reshape d_concat → d_out_mh: (d_head, seq_len, num_heads)
        Matrix<float> d_out_mh(Dh, seq_len, H);
        for (int h = 0; h < H; ++h) {
            for (int d = 0; d < Dh; ++d) {
                int src_row = h * Dh + d;
                for (int t = 0; t < seq_len; ++t)
                    d_out_mh.at(d, t, h) = d_concat.at(src_row, t);
            }
        }

        // Build Q_mh, K_mh, V_mh from saved raw tensors (for backward computation)
        Matrix<float> Q_mh(Dh, seq_len, H);
        Matrix<float> K_mh(Dh, seq_len, H);
        Matrix<float> V_mh(Dh, seq_len, H);
        for (int h = 0; h < H; ++h) {
            for (int d = 0; d < Dh; ++d) {
                int src_row = h * Dh + d;
                for (int t = 0; t < seq_len; ++t) {
                    Q_mh.at(d, t, h) = Q_raw.at(src_row, t);
                    K_mh.at(d, t, h) = K_raw.at(src_row, t);
                    V_mh.at(d, t, h) = V_raw.at(src_row, t);
                }
            }
        }

        Matrix<float> dQ_mh(Dh, seq_len, H);
        Matrix<float> dK_mh(Dh, seq_len, H);
        Matrix<float> dV_mh(Dh, seq_len, H);

        for (int h = 0; h < H; ++h) {
            Matrix<float> Q_h = Q_mh.getChannel(h);      // (d_head, seq_len)
            Matrix<float> V_h = V_mh.getChannel(h);      // (d_head, seq_len)
            Matrix<float> d_out_h = d_out_mh.getChannel(h);  // (d_head, seq_len)

            // dV = d_out * attn^T: (d_head, seq_len) * (seq_len, seq_len) = (d_head, seq_len)
            // (out = V * attn → dL/dV = dL/dout * attn^T)
            // Build attn_h: (seq_len, seq_len) matrix from attn_w channel h
            Matrix<float> attn_h(seq_len, seq_len);
            for (int i = 0; i < seq_len; ++i)
                for (int j = 0; j < seq_len; ++j)
                    attn_h.at(i, j) = attn_w.at(i, j, h);

            Matrix<float> attn_h_T = attn_h.Translate();  // (seq_len, seq_len)
            Matrix<float> dV_h = d_out_h * attn_h_T;       // (d_head, seq_len) * (seq_len, seq_len) = (d_head, seq_len)

            for (int d = 0; d < Dh; ++d)
                for (int t = 0; t < seq_len; ++t)
                    dV_mh.at(d, t, h) = dV_h.at(d, t);

            // d_attn = V_h^T * d_out_h: (seq_len, d_head) * (d_head, seq_len) = (seq_len, seq_len)
            Matrix<float> V_h_T = V_h.Translate();  // (seq_len, d_head)
            Matrix<float> d_attn_raw = V_h_T * d_out_h;  // (seq_len, seq_len)

            // Softmax backward: d_scores = attn * (d_attn - rowsum(attn * d_attn))
            Matrix<float> d_scores(seq_len, seq_len);
            for (int i = 0; i < seq_len; ++i) {
                // Compute Σ_k attn(i,k) * d_attn(i,k) for this row
                float row_sum = 0.0f;
                for (int k = 0; k < seq_len; ++k)
                    row_sum += attn_h.at(i, k) * d_attn_raw.at(i, k);

                for (int j = 0; j < seq_len; ++j) {
                    float grad = attn_h.at(i, j) * (d_attn_raw.at(i, j) - row_sum);
                    // Apply scale factor that was used in forward
                    d_scores.at(i, j) = grad * scale;
                }
            }

            // dK = Q * d_scores^T: (d_head, seq_len) * (seq_len, seq_len) = (d_head, seq_len)
            // (scores = K^T * Q → ∂L/∂K = Q * (∂L/∂scores)^T)
            Matrix<float> d_scores_T = d_scores.Translate();
            Matrix<float> dK_h = Q_h * d_scores_T;       // (d_head, seq_len) * (seq_len, seq_len) = (d_head, seq_len)

            // dQ = K * d_scores: (d_head, seq_len) * (seq_len, seq_len) = (d_head, seq_len)
            // (scores = K^T * Q → ∂L/∂Q = K * ∂L/∂scores)
            Matrix<float> K_h = K_mh.getChannel(h);      // (d_head, seq_len)
            Matrix<float> dQ_h = K_h * d_scores;          // (d_head, seq_len) * (seq_len, seq_len) = (d_head, seq_len)

            for (int d = 0; d < Dh; ++d) {
                for (int t = 0; t < seq_len; ++t) {
                    dK_mh.at(d, t, h) = dK_h.at(d, t);
                    dQ_mh.at(d, t, h) = dQ_h.at(d, t);
                }
            }
        }

        // --- 3. Reshape dQ/dK/dV from (d_head, seq_len, num_heads) → (d_model, seq_len) ---
        Matrix<float> dQ_raw(D, seq_len);
        Matrix<float> dK_raw(D, seq_len);
        Matrix<float> dV_raw(D, seq_len);
        for (int h = 0; h < H; ++h) {
            for (int d = 0; d < Dh; ++d) {
                int dst_row = h * Dh + d;
                for (int t = 0; t < seq_len; ++t) {
                    dQ_raw.at(dst_row, t) = dQ_mh.at(d, t, h);
                    dK_raw.at(dst_row, t) = dK_mh.at(d, t, h);
                    dV_raw.at(dst_row, t) = dV_mh.at(d, t, h);
                }
            }
        }

        // --- 4. Gradient through Q/K/V linear projections ---
        // dW_q = dQ_raw * input: (d_model, seq_len) * (seq_len, d_model) = (d_model, d_model)
        dW_q = new Matrix<float>(D, D);
        Matrix<float> dW_q_tmp = dQ_raw * input;
        for (int i = 0; i < D; ++i)
            for (int j = 0; j < D; ++j)
                dW_q->at(i, j) = dW_q_tmp.at(i, j);

        dW_k = new Matrix<float>(D, D);
        Matrix<float> dW_k_tmp = dK_raw * input;
        for (int i = 0; i < D; ++i)
            for (int j = 0; j < D; ++j)
                dW_k->at(i, j) = dW_k_tmp.at(i, j);

        dW_v = new Matrix<float>(D, D);
        Matrix<float> dW_v_tmp = dV_raw * input;
        for (int i = 0; i < D; ++i)
            for (int j = 0; j < D; ++j)
                dW_v->at(i, j) = dW_v_tmp.at(i, j);

        // Bias gradients: column-sum of dQ/dK/dV_raw
        db_q = new Matrix<float>(D, 1);
        db_k = new Matrix<float>(D, 1);
        db_v = new Matrix<float>(D, 1);
        for (int d = 0; d < D; ++d) {
            float sq = 0, sk = 0, sv = 0;
            for (int t = 0; t < seq_len; ++t) {
                sq += dQ_raw.at(d, t);
                sk += dK_raw.at(d, t);
                sv += dV_raw.at(d, t);
            }
            db_q->at(d, 0) = sq;
            db_k->at(d, 0) = sk;
            db_v->at(d, 0) = sv;
        }

        // --- 5. Gradient w.r.t. input ---
        // d_input = W_q^T * dQ_raw + W_k^T * dK_raw + W_v^T * dV_raw
        Matrix<float> W_q_T = W_q->Translate();
        Matrix<float> W_k_T = W_k->Translate();
        Matrix<float> W_v_T = W_v->Translate();

        Matrix<float> d_input_T = (W_q_T * dQ_raw) + (W_k_T * dK_raw) + (W_v_T * dV_raw);
        // d_input_T is (d_model, seq_len), transpose to (seq_len, d_model)
        Matrix<float> d_input(seq_len, D);
        for (int t = 0; t < seq_len; ++t)
            for (int d = 0; d < D; ++d)
                d_input.at(t, d) = d_input_T.at(d, t);

        return d_input;
    }

    // ===============================[Parameter Access]===============================
    std::vector<Matrix<float>*> getParams() override {
        return {W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o};
    }

    std::vector<Matrix<float>*> getAllGrads() override {
        return {dW_q, dW_k, dW_v, dW_o, db_q, db_k, db_v, db_o};
    }

    Matrix<float>* getGard() override { return gard.empty() ? nullptr : gard.back(); }
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }

    void CleanGard() override {
        for (auto p : gard) delete p;
        gard.clear();
        for (auto p : output) delete p;
        output.clear();
        delete dW_q; dW_q = nullptr;
        delete dW_k; dW_k = nullptr;
        delete dW_v; dW_v = nullptr;
        delete dW_o; dW_o = nullptr;
        delete db_q; db_q = nullptr;
        delete db_k; db_k = nullptr;
        delete db_v; db_v = nullptr;
        delete db_o; db_o = nullptr;
    }
};

} // namespace CoreNNSpace

#endif // COREPP_LAYERS_MULTIHEADATTENTION_H
