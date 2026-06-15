#include "Layers/Attention/TransformerEncoder.h"

namespace NPCore {

// =================================[TransformerEncoderLayer]================================

TransformerEncoderLayer::TransformerEncoderLayer(int d_model, int nhead, int dim_ff)
    : d_model(d_model)
{
    mha   = new MultiHeadAttention(d_model, nhead);
    norm1 = new LayerNorm(d_model);
    norm2 = new LayerNorm(d_model);
    fc1   = new Linear(d_model, dim_ff);
    fc2   = new Linear(dim_ff, d_model);
    gelu  = new Activation::GELU();
}

TransformerEncoderLayer::~TransformerEncoderLayer() {
    delete mha; delete norm1; delete norm2;
    delete fc1; delete fc2; delete gelu;
    delete mha_out; delete ffn_out;
}

Matrix<float> TransformerEncoderLayer::forward(Matrix<float>& input) {
    // Pre-norm architecture
    int seq_len = input.row;

    // Step 1: Self-attention sublayer
    Matrix<float> norm1_out = norm1->forward(const_cast<Matrix<float>&>(input));
    Matrix<float> attn_out  = mha->forward(norm1_out);
    delete mha_out;
    mha_out = new Matrix<float>(input + attn_out);

    // Step 2: Feed-forward sublayer
    // Linear(bias) is (out_f, 1) which doesn't broadcast to (out_f, seq_len).
    // Use manual GEMM per-position: for each token, y_i = W2 * GELU(W1 * x_i + b1) + b2
    Matrix<float> norm2_out = norm2->forward(*mha_out); // (seq_len, d_model)

    auto W1 = fc1->getParams(); auto b1 = W1.size()>1 ? W1[1] : nullptr; auto w1 = W1[0];
    auto W2 = fc2->getParams(); auto b2 = W2.size()>1 ? W2[1] : nullptr; auto w2 = W2[0];
    int dim_ff = w1->row;

    auto* fc1_out = new Matrix<float>(seq_len, dim_ff);
    auto* act_out_ptr = new Matrix<float>(seq_len, dim_ff);
    auto* fc2_out = new Matrix<float>(seq_len, d_model);

    for (int t = 0; t < seq_len; ++t) {
        // Extract token t as column vector
        Matrix<float> x_t(d_model, 1);
        for (int d = 0; d < d_model; ++d) x_t.at(d, 0) = norm2_out.at(t, d);

        // FC1: W1 * x_t + b1
        Matrix<float> h = (*w1) * x_t;
        if (b1) for (int d = 0; d < dim_ff; ++d) h.at(d, 0) += b1->at(d, 0);

        // Save h for backward
        for (int d = 0; d < dim_ff; ++d) fc1_out->at(t, d) = h.at(d, 0);

        // GELU
        Matrix<float> act(dim_ff, 1);
        for (int d = 0; d < dim_ff; ++d) act.at(d, 0) = h.at(d, 0);
        Matrix<float> gelu_out = gelu->forward(act);

        for (int d = 0; d < dim_ff; ++d) act_out_ptr->at(t, d) = gelu_out.at(d, 0);

        // FC2: W2 * gelu_out + b2
        Matrix<float> y = (*w2) * gelu_out;
        if (b2) for (int d = 0; d < d_model; ++d) y.at(d, 0) += b2->at(d, 0);

        for (int d = 0; d < d_model; ++d) fc2_out->at(t, d) = y.at(d, 0);
    }

    delete ffn_out;
    ffn_out = new Matrix<float>(*mha_out + (*fc2_out));

    if (train_mode) {
        gard.push_back(new Matrix<float>(input));
        gard.push_back(new Matrix<float>(norm2_out));
        gard.push_back(fc1_out);        // pre-activation values
        gard.push_back(act_out_ptr);    // post-activation values
        gard.push_back(new Matrix<float>(*fc2_out)); // FFN output (for residual grad)
    } else {
        delete fc1_out; delete act_out_ptr; delete fc2_out;
    }

    auto* result = new Matrix<float>(*ffn_out);
    output.push_back(result);
    return *result;
}

Matrix<float> TransformerEncoderLayer::backward(Matrix<float>& grad_output) {
    int seq_len = grad_output.row;
    auto W1 = fc1->getParams(); auto w1 = W1[0]; int dim_ff = w1->row;
    auto W2 = fc2->getParams(); auto w2 = W2[0];

    // Run fc1/fc2 backward once to allocate gradient storage
    fc1->CleanGard(); fc2->CleanGard();
    { Matrix<float> dx(d_model,1); fc1->forward(dx); Matrix<float> dg(dim_ff,1); fc1->backward(dg); }
    { Matrix<float> dx(dim_ff,1); fc2->forward(dx); Matrix<float> dg(d_model,1); fc2->backward(dg); }

    // gard: [0]=input, [1]=norm2_out, [2]=fc1_out, [3]=act_out, [4]=fc2_out
    Matrix<float>& norm2_saved = *gard[gard.size()-4];
    Matrix<float>& act_saved   = *gard[gard.size()-2];

    Matrix<float> d_mha_out(seq_len, d_model);
    Matrix<float> d_norm2_acc(seq_len, d_model);

    auto *dW1 = fc1->getWeightGrad(), *db1 = fc1->getBiasGrad();
    auto *dW2 = fc2->getWeightGrad(), *db2 = fc2->getBiasGrad();

    // Zero dummy gradients
    for (int i = 0; i < dim_ff*d_model; ++i) dW1->data_ptr()[i] = 0;
    if (db1) for (int i = 0; i < dim_ff; ++i) db1->data_ptr()[i] = 0;
    for (int i = 0; i < d_model*dim_ff; ++i) dW2->data_ptr()[i] = 0;
    if (db2) for (int i = 0; i < d_model; ++i) db2->data_ptr()[i] = 0;

    for (int t = 0; t < seq_len; ++t) {
        Matrix<float> d_fc2_t(d_model, 1);
        for (int d = 0; d < d_model; ++d) { d_fc2_t.at(d, 0) = grad_output.at(t, d); d_mha_out.at(t, d) = grad_output.at(t, d); }

        for (int i = 0; i < d_model; ++i)
            for (int j = 0; j < dim_ff; ++j)
                dW2->at(i, j) += d_fc2_t.at(i, 0) * act_saved.at(t, j);
        if (db2) for (int i = 0; i < d_model; ++i) db2->at(i, 0) += d_fc2_t.at(i, 0);

        Matrix<float> d_gelu = w2->Translate() * d_fc2_t;

        Matrix<float> act_col(dim_ff, 1);
        for (int d = 0; d < dim_ff; ++d) act_col.at(d, 0) = act_saved.at(t, d);
        Matrix<float> d_pre_gelu = gelu->backward(act_col);
        for (int d = 0; d < dim_ff; ++d) d_pre_gelu.at(d, 0) *= d_gelu.at(d, 0);

        for (int i = 0; i < dim_ff; ++i)
            for (int j = 0; j < d_model; ++j)
                dW1->at(i, j) += d_pre_gelu.at(i, 0) * norm2_saved.at(t, j);
        if (db1) for (int i = 0; i < dim_ff; ++i) db1->at(i, 0) += d_pre_gelu.at(i, 0);

        Matrix<float> d_x = w1->Translate() * d_pre_gelu;
        for (int d = 0; d < d_model; ++d) d_norm2_acc.at(t, d) = d_x.at(d, 0);
    }

    Matrix<float> d_norm2 = norm2->backward(d_norm2_acc);
    for (int i = 0; i < seq_len; ++i)
        for (int j = 0; j < d_model; ++j)
            d_mha_out.at(i, j) += d_norm2.at(i, j);

    Matrix<float> d_attn = mha->backward(d_mha_out);
    Matrix<float> d_norm1 = norm1->backward(d_attn);

    Matrix<float> d_input(seq_len, d_model);
    for (int i = 0; i < seq_len; ++i)
        for (int j = 0; j < d_model; ++j)
            d_input.at(i, j) = d_norm1.at(i, j) + d_mha_out.at(i, j);

    return d_input;
}

void TransformerEncoderLayer::CleanGard() {
    for (auto p : gard) { delete p; }
    gard.clear();
    for (auto p : output) { delete p; }
    output.clear();
    mha->CleanGard();
    norm1->CleanGard();
    norm2->CleanGard();
    fc1->CleanGard();
    fc2->CleanGard();
    gelu->CleanGard();
    delete mha_out; mha_out = nullptr;
    delete ffn_out; ffn_out = nullptr;
}

std::vector<Matrix<float>*> TransformerEncoderLayer::getParams() {
    auto p = mha->getParams();
    for (auto* m : norm1->getParams()) p.push_back(m);
    for (auto* m : norm2->getParams()) p.push_back(m);
    for (auto* m : fc1->getParams()) p.push_back(m);
    for (auto* m : fc2->getParams()) p.push_back(m);
    return p;
}

std::vector<Matrix<float>*> TransformerEncoderLayer::getAllGrads() {
    auto g = mha->getAllGrads();
    for (auto* m : norm1->getAllGrads()) g.push_back(m);
    for (auto* m : norm2->getAllGrads()) g.push_back(m);
    for (auto* m : fc1->getAllGrads()) g.push_back(m);
    for (auto* m : fc2->getAllGrads()) g.push_back(m);
    return g;
}

Matrix<float>* TransformerEncoderLayer::getGard() { return nullptr; }
Matrix<float>* TransformerEncoderLayer::getOutput() {
    return output.empty() ? nullptr : output.back();
}

void TransformerEncoderLayer::cuda() {
    mha->cuda(); norm1->cuda(); norm2->cuda();
    fc1->cuda(); fc2->cuda(); gelu->cuda();
}
void TransformerEncoderLayer::cpu() {
    mha->cpu(); norm1->cpu(); norm2->cpu();
    fc1->cpu(); fc2->cpu(); gelu->cpu();
}
void TransformerEncoderLayer::eval() {
    train_mode = false;
    mha->eval(); norm1->eval(); norm2->eval();
    fc1->eval(); fc2->eval(); gelu->eval();
}
void TransformerEncoderLayer::train() {
    train_mode = true;
    mha->train(); norm1->train(); norm2->train();
    fc1->train(); fc2->train(); gelu->train();
}

// =================================[TransformerEncoder]================================

TransformerEncoder::TransformerEncoder(int d_model, int nhead, int dim_ff, int num_layers) {
    for (int i = 0; i < num_layers; ++i)
        layers.push_back(new TransformerEncoderLayer(d_model, nhead, dim_ff));
}

TransformerEncoder::~TransformerEncoder() {
    for (auto* L : layers) delete L;
}

Matrix<float> TransformerEncoder::forward(Matrix<float>& input) {
    Matrix<float> out = input;
    for (auto* layer : layers) out = layer->forward(out);
    auto* result = new Matrix<float>(out);
    output.push_back(result);
    return *result;
}

Matrix<float> TransformerEncoder::backward(Matrix<float>& grad_output) {
    Matrix<float> g = grad_output;
    for (int i = (int)layers.size() - 1; i >= 0; --i)
        g = layers[i]->backward(g);
    return g;
}

void TransformerEncoder::CleanGard() {
    for (auto p : gard) { delete p; }
    gard.clear();
    for (auto p : output) { delete p; }
    output.clear();
    for (auto* L : layers) L->CleanGard();
}

std::vector<Matrix<float>*> TransformerEncoder::getParams() {
    std::vector<Matrix<float>*> all;
    for (auto* L : layers) {
        auto p = L->getParams();
        all.insert(all.end(), p.begin(), p.end());
    }
    return all;
}

std::vector<Matrix<float>*> TransformerEncoder::getAllGrads() {
    std::vector<Matrix<float>*> all;
    for (auto* L : layers) {
        auto g = L->getAllGrads();
        all.insert(all.end(), g.begin(), g.end());
    }
    return all;
}

Matrix<float>* TransformerEncoder::getGard() { return nullptr; }
Matrix<float>* TransformerEncoder::getOutput() {
    return output.empty() ? nullptr : output.back();
}

void TransformerEncoder::cuda() { for (auto* L : layers) L->cuda(); }
void TransformerEncoder::cpu()  { for (auto* L : layers) L->cpu();  }
void TransformerEncoder::eval()  {
    train_mode = false;
    for (auto* L : layers) L->eval();
}
void TransformerEncoder::train() {
    train_mode = true;
    for (auto* L : layers) L->train();
}

} // namespace NPCore
