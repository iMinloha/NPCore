#ifndef NPCORE_LAYERS_EMBEDDING_H
#define NPCORE_LAYERS_EMBEDDING_H

#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace NPCore {

class Embedding : public Module<float> {
    Matrix<float>* weight;
    int vocab_size, embed_dim;
    Matrix<float>* dW = nullptr;

public:
    Embedding(int vocab_size, int embed_dim);
    ~Embedding() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override { return {weight}; }
    std::vector<Matrix<float>*> getAllGrads() override { return {dW}; }
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }
};

} // namespace NPCore
#endif
