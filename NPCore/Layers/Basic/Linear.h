#ifndef NPCORE_LAYERS_LINEAR_H
#define NPCORE_LAYERS_LINEAR_H

#include <vector>
#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace NPCore {

// =================================[Linear Fully-Connected Layer]================================
// y = W * x + b

class NPCORE_API Linear : public Module<float> {
private:
    Matrix<float>* weight;
    Matrix<float>* bias;

public:
    Linear(int in_features, int out_features, InitMode mode = InitMode::Uniform,
           double mu = 0.0, double sigma = 1.0);

    ~Linear() override;

    Matrix<float> forward(Matrix<float> &input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;

    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
    void CleanGard() override;
};

} // namespace NPCore

#endif // NPCORE_LAYERS_LINEAR_H
