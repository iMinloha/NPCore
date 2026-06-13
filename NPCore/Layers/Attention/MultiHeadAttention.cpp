#include "Layers/Attention/MultiHeadAttention.h"
#include <cmath>
#include "Core/Assert.h"

namespace NPCore {

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads, bool causal)
    : d_model(d_model), num_heads(num_heads), d_head(d_model / num_heads),
      scale(1.0f / std::sqrt(static_cast<float>(d_head))), causal(causal)
{
    NPCORE_ASSERT(d_model % num_heads == 0,
                "d_model (%d) must be divisible by num_heads (%d)", d_model, num_heads);

    W_q = new Matrix<float>(d_model, d_model);
    W_k = new Matrix<float>(d_model, d_model);
    W_v = new Matrix<float>(d_model, d_model);
    W_o = new Matrix<float>(d_model, d_model);
    b_q = new Matrix<float>(d_model, 1);
    b_k = new Matrix<float>(d_model, 1);
    b_v = new Matrix<float>(d_model, 1);
    b_o = new Matrix<float>(d_model, 1);

    InitMatrixFunc(*W_q, InitMode::XavierUniform);
    InitMatrixFunc(*W_k, InitMode::XavierUniform);
    InitMatrixFunc(*W_v, InitMode::XavierUniform);
    InitMatrixFunc(*W_o, InitMode::XavierUniform);
    InitMatrixFunc(*b_q, InitMode::Zeros);
    InitMatrixFunc(*b_k, InitMode::Zeros);
    InitMatrixFunc(*b_v, InitMode::Zeros);
    InitMatrixFunc(*b_o, InitMode::Zeros);
}

MultiHeadAttention::~MultiHeadAttention() {
    delete W_q; delete W_k; delete W_v; delete W_o;
    delete b_q; delete b_k; delete b_v; delete b_o;
    delete dW_q; delete dW_k; delete dW_v; delete dW_o;
    delete db_q; delete db_k; delete db_v; delete db_o;
    delete Q_saved; delete K_saved; delete V_saved;
    delete attn_saved; delete concat_saved;
}

Matrix<float> MultiHeadAttention::forward(Matrix<float>& input) {
    const int seq_len = input.row;
    const int D = d_model, H = num_heads, Dh = d_head;

    Matrix<float> input_T = input.Translate();
    Matrix<float> Q_raw = (*W_q) * input_T;
    Matrix<float> K_raw = (*W_k) * input_T;
    Matrix<float> V_raw = (*W_v) * input_T;

    for (int d = 0; d < D; ++d) {
        float bq = b_q->at(d, 0), bk = b_k->at(d, 0), bv = b_v->at(d, 0);
        for (int t = 0; t < seq_len; ++t) {
            Q_raw.at(d, t) += bq;
            K_raw.at(d, t) += bk;
            V_raw.at(d, t) += bv;
        }
    }

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

    Matrix<float> out_mh(Dh, seq_len, H);
    Matrix<float> attn_weights(seq_len, seq_len, H);

    for (int h = 0; h < H; ++h) {
        Matrix<float> K_h_T = K_mh.getChannel(h).Translate();
        Matrix<float> Q_h = Q_mh.getChannel(h);
        Matrix<float> scores = K_h_T * Q_h;

        for (int i = 0; i < seq_len; ++i)
            for (int j = 0; j < seq_len; ++j)
                scores.at(i, j) *= scale;

        if (causal) {
            for (int i = 0; i < seq_len; ++i)
                for (int j = i + 1; j < seq_len; ++j)
                    scores.at(i, j) = -1e30f;
        }

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

        for (int i = 0; i < seq_len; ++i)
            for (int j = 0; j < seq_len; ++j)
                attn_weights.at(i, j, h) = scores.at(i, j);

        Matrix<float> V_h = V_mh.getChannel(h);
        Matrix<float> out_h = V_h * scores;
        for (int d = 0; d < Dh; ++d)
            for (int t = 0; t < seq_len; ++t)
                out_mh.at(d, t, h) = out_h.at(d, t);
    }

    Matrix<float> concat(D, seq_len);
    for (int h = 0; h < H; ++h) {
        for (int d = 0; d < Dh; ++d) {
            int dst_row = h * Dh + d;
            for (int t = 0; t < seq_len; ++t)
                concat.at(dst_row, t) = out_mh.at(d, t, h);
        }
    }

    Matrix<float> result_raw = (*W_o) * concat;
    for (int d = 0; d < D; ++d)
        for (int t = 0; t < seq_len; ++t)
            result_raw.at(d, t) += b_o->at(d, 0);

    auto* out = new Matrix<float>(seq_len, D);
    for (int t = 0; t < seq_len; ++t)
        for (int d = 0; d < D; ++d)
            out->at(t, d) = result_raw.at(d, t);

    if (train_mode) {
        gard.push_back(new Matrix<float>(input));
        delete Q_saved; Q_saved = new Matrix<float>(Q_raw);
        delete K_saved; K_saved = new Matrix<float>(K_raw);
        delete V_saved; V_saved = new Matrix<float>(V_raw);
        delete attn_saved; attn_saved = new Matrix<float>(attn_weights);
        delete concat_saved; concat_saved = new Matrix<float>(concat);
    }

    output.push_back(out);
    return *out;
}

Matrix<float> MultiHeadAttention::backward(Matrix<float>& grad_output) {
    const int seq_len = grad_output.row;
    const int D = d_model, H = num_heads, Dh = d_head;

    Matrix<float>& input = *gard.back();
    Matrix<float>& Q_raw = *Q_saved;
    Matrix<float>& K_raw = *K_saved;
    Matrix<float>& V_raw = *V_saved;
    Matrix<float>& attn_w = *attn_saved;
    Matrix<float>& concat = *concat_saved;

    Matrix<float> d_result(D, seq_len);
    for (int t = 0; t < seq_len; ++t)
        for (int d = 0; d < D; ++d)
            d_result.at(d, t) = grad_output.at(t, d);

    dW_o = new Matrix<float>(D, D);
    Matrix<float> concat_T = concat.Translate();
    Matrix<float> dW_o_tmp = d_result * concat_T;
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < D; ++j)
            dW_o->at(i, j) = dW_o_tmp.at(i, j);

    db_o = new Matrix<float>(D, 1);
    for (int d = 0; d < D; ++d) {
        float s = 0;
        for (int t = 0; t < seq_len; ++t) s += d_result.at(d, t);
        db_o->at(d, 0) = s;
    }

    Matrix<float> W_o_T = W_o->Translate();
    Matrix<float> d_concat = W_o_T * d_result;

    Matrix<float> d_out_mh(Dh, seq_len, H);
    for (int h = 0; h < H; ++h) {
        for (int d = 0; d < Dh; ++d) {
            int src_row = h * Dh + d;
            for (int t = 0; t < seq_len; ++t)
                d_out_mh.at(d, t, h) = d_concat.at(src_row, t);
        }
    }

    Matrix<float> Q_mh(Dh, seq_len, H), K_mh(Dh, seq_len, H), V_mh(Dh, seq_len, H);
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

    Matrix<float> dQ_mh(Dh, seq_len, H), dK_mh(Dh, seq_len, H), dV_mh(Dh, seq_len, H);

    for (int h = 0; h < H; ++h) {
        Matrix<float> Q_h = Q_mh.getChannel(h);
        Matrix<float> V_h = V_mh.getChannel(h);
        Matrix<float> d_out_h = d_out_mh.getChannel(h);

        Matrix<float> attn_h(seq_len, seq_len);
        for (int i = 0; i < seq_len; ++i)
            for (int j = 0; j < seq_len; ++j)
                attn_h.at(i, j) = attn_w.at(i, j, h);

        Matrix<float> attn_h_T = attn_h.Translate();
        Matrix<float> dV_h = d_out_h * attn_h_T;
        for (int d = 0; d < Dh; ++d)
            for (int t = 0; t < seq_len; ++t)
                dV_mh.at(d, t, h) = dV_h.at(d, t);

        Matrix<float> V_h_T = V_h.Translate();
        Matrix<float> d_attn_raw = V_h_T * d_out_h;

        Matrix<float> d_scores(seq_len, seq_len);
        for (int i = 0; i < seq_len; ++i) {
            float row_sum = 0.0f;
            for (int k = 0; k < seq_len; ++k)
                row_sum += attn_h.at(i, k) * d_attn_raw.at(i, k);
            for (int j = 0; j < seq_len; ++j) {
                float grad = attn_h.at(i, j) * (d_attn_raw.at(i, j) - row_sum);
                d_scores.at(i, j) = grad * scale;
            }
        }

        Matrix<float> d_scores_T = d_scores.Translate();
        Matrix<float> dK_h = Q_h * d_scores_T;
        Matrix<float> K_h = K_mh.getChannel(h);
        Matrix<float> dQ_h = K_h * d_scores;

        for (int d = 0; d < Dh; ++d) {
            for (int t = 0; t < seq_len; ++t) {
                dK_mh.at(d, t, h) = dK_h.at(d, t);
                dQ_mh.at(d, t, h) = dQ_h.at(d, t);
            }
        }
    }

    Matrix<float> dQ_raw(D, seq_len), dK_raw(D, seq_len), dV_raw(D, seq_len);
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

    Matrix<float> W_q_T = W_q->Translate();
    Matrix<float> W_k_T = W_k->Translate();
    Matrix<float> W_v_T = W_v->Translate();
    Matrix<float> d_input_T = (W_q_T * dQ_raw) + (W_k_T * dK_raw) + (W_v_T * dV_raw);
    Matrix<float> d_input(seq_len, D);
    for (int t = 0; t < seq_len; ++t)
        for (int d = 0; d < D; ++d)
            d_input.at(t, d) = d_input_T.at(d, t);

    return d_input;
}

void MultiHeadAttention::CleanGard() {
    for (auto p : gard) { delete p; }
    gard.clear();
    for (auto p : output) { delete p; }
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

} // namespace NPCore
