#include "Layers/Basic/Embedding.h"
#include "Core/Assert.h"

namespace NPCore {

Embedding::Embedding(int vocab_size, int embed_dim)
    : vocab_size(vocab_size), embed_dim(embed_dim) {
    weight = new Matrix<float>(vocab_size, embed_dim);
    InitMatrixFunc(*weight, Gaussian, {.mu=0,.sigma=1.0f/std::sqrt((float)embed_dim)});
}

Embedding::~Embedding() { delete weight; delete dW; }

Matrix<float> Embedding::forward(Matrix<float>& input) {
    int seq = input.row;
    auto* out = new Matrix<float>(seq, embed_dim);
    for (int t = 0; t < seq; ++t) {
        int idx = (int)input.at(t,0);
        NPCORE_ASSERT(idx >= 0 && idx < vocab_size, "Embedding index out of range");
        for (int e = 0; e < embed_dim; ++e) out->at(t,e) = weight->at(idx,e);
    }
    if (train_mode) gard.push_back(new Matrix<float>(input));
    output.push_back(out); return *out;
}

Matrix<float> Embedding::backward(Matrix<float>& grad_output) {
    dW = new Matrix<float>(vocab_size, embed_dim);
    auto& indices = *gard.back();
    for (int t = 0; t < grad_output.row; ++t) {
        int idx = (int)indices.at(t,0);
        for (int e = 0; e < embed_dim; ++e) dW->at(idx,e) += grad_output.at(t,e);
    }
    return Matrix<float>(grad_output.row, grad_output.col);
}

void Embedding::CleanGard() {
    for (auto p : gard) { delete p; }
    gard.clear();
    for (auto p : output) { delete p; }
    output.clear();
    delete dW; dW=nullptr;
}

std::vector<Matrix<float>*> Embedding::getParams() { return {weight}; }
std::vector<Matrix<float>*> Embedding::getAllGrads() { return {dW}; }
Matrix<float>* Embedding::getGard() { return nullptr; }
Matrix<float>* Embedding::getOutput() { return output.empty() ? nullptr : output.back(); }

} // namespace NPCore
