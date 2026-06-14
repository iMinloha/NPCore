#ifndef NPCORE_LAYERS_DROPOUT_H
#define NPCORE_LAYERS_DROPOUT_H

#include "Layers/Module.h"

namespace NPCore {

class NPCORE_API Dropout : public Module<float> {
    float rate;
    Matrix<float>* mask_ = nullptr;

public:
    explicit Dropout(float rate = 0.5f);
    ~Dropout() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
};

} // namespace NPCore
#endif
