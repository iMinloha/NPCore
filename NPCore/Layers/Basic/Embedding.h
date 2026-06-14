#ifndef NPCORE_LAYERS_EMBEDDING_H
#define NPCORE_LAYERS_EMBEDDING_H

#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace NPCore {

class NPCORE_API Embedding : public Module<float> {
    Matrix<float>* weight;
    int vocab_size, embed_dim;
    Matrix<float>* dW = nullptr;

public:
    Embedding(int vocab_size, int embed_dim);
    ~Embedding() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override;
    std::vector<Matrix<float>*> getAllGrads() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
};

} // namespace NPCore
#endif
