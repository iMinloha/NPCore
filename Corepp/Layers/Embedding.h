#ifndef COREPP_LAYERS_EMBEDDING_H
#define COREPP_LAYERS_EMBEDDING_H
#include "Layers/Module.h"
#include "Layers/ParamInit.h"
namespace CoreNNSpace {

class Embedding : public Module<float> {
    Matrix<float>* weight; // (vocab_size, embed_dim)
    int vocab_size, embed_dim;
    Matrix<float>* dW = nullptr;

public:
    Embedding(int vocab_size, int embed_dim) : vocab_size(vocab_size), embed_dim(embed_dim) {
        weight = new Matrix<float>(vocab_size, embed_dim);
        InitMatrixFunc(*weight, Gaussian, {.mu=0,.sigma=1.0f/std::sqrt((float)embed_dim)});
    }
    ~Embedding() override { delete weight; delete dW; }

    Matrix<float> forward(Matrix<float>& input) override {
        // input: (seq_len, 1) integer indices
        // output: (seq_len, embed_dim)
        int seq = input.row;
        auto* out = new Matrix<float>(seq, embed_dim);
        for (int t = 0; t < seq; ++t) {
            int idx = (int)input.at(t,0);
            COREPP_ASSERT(idx >= 0 && idx < vocab_size, "Embedding index out of range");
            for (int e = 0; e < embed_dim; ++e) out->at(t,e) = weight->at(idx,e);
        }
        if (train_mode) gard.push_back(new Matrix<float>(input));
        output.push_back(out); return *out;
    }

    Matrix<float> backward(Matrix<float>& grad_output) override {
        dW = new Matrix<float>(vocab_size, embed_dim);
        auto& indices = *gard.back();
        for (int t = 0; t < grad_output.row; ++t) {
            int idx = (int)indices.at(t,0);
            for (int e = 0; e < embed_dim; ++e) dW->at(idx,e) += grad_output.at(t,e);
        }
        return Matrix<float>(grad_output.row, grad_output.col); // dx = 0 (indices are discrete)
    }

    std::vector<Matrix<float>*> getParams() override { return {weight}; }
    std::vector<Matrix<float>*> getAllGrads() override { return {dW}; }
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override { for(auto p:gard)delete p; gard.clear(); for(auto p:output)delete p; output.clear();
                                 delete dW; dW=nullptr; }
};
} // namespace
#endif
