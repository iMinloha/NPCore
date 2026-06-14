#ifndef NPCORE_ACTIVATIONS_H
#define NPCORE_ACTIVATIONS_H

#include <vector>
#include <cmath>
#include "Layers/Module.h"

namespace NPCore {
namespace Activation {

// =================================[ActivationLayer — shared base for all activations]================================
// Provides the four boilerplate methods shared by all 11 activation classes.
// Subclasses only need to implement forward() and backward().

class NPCORE_API ActivationLayer : public Module<float> {
public:
    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
    void CleanGard() override;
};

// ============ ReLU ============
class NPCORE_API ReLU : public ActivationLayer {
public:
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
};

// ============ LeakyReLU ============
class NPCORE_API LeakyReLU : public ActivationLayer {
    float alpha;
public:
    LeakyReLU(float a = 0.01f);
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
};

// ============ Sigmoid ============
class NPCORE_API Sigmoid : public ActivationLayer {
public:
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
};

// ============ Tanh ============
class NPCORE_API Tanh : public ActivationLayer {
public:
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
};

// ============ SoftMax ============
class NPCORE_API SoftMax : public ActivationLayer {
public:
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
};

// ============ ELU ============
class NPCORE_API ELU : public ActivationLayer {
    float alpha;
public:
    ELU(float a = 1.0f);
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
};

// ============ SELU ============
class NPCORE_API SELU : public ActivationLayer {
public:
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
};

// ============ Softplus ============
class NPCORE_API Softplus : public ActivationLayer {
public:
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
};

// ============ Mish ============
class NPCORE_API Mish : public ActivationLayer {
public:
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
};

// ============ GELU ============
class NPCORE_API GELU : public ActivationLayer {
public:
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
};

// ============ Swish ============
class NPCORE_API Swish : public ActivationLayer {
public:
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
};

} // namespace Activation
} // namespace NPCore
#endif
