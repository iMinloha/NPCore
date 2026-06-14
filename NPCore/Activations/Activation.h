#ifndef NPCORE_ACTIVATIONS_H
#define NPCORE_ACTIVATIONS_H

#include <vector>
#include <cmath>
#include "Layers/Module.h"

namespace NPCore {
namespace Activation {

// ============ ReLU ============
class ReLU : public Module<float> {
public:
    ReLU() = default; ~ReLU() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
    void CleanGard() override;
};

// ============ LeakyReLU ============
class LeakyReLU : public Module<float> {
    float alpha;
public:
    LeakyReLU(float a = 0.01f); ~LeakyReLU() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
    void CleanGard() override;
};

// ============ Sigmoid ============
class Sigmoid : public Module<float> {
public:
    Sigmoid() = default; ~Sigmoid() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
    void CleanGard() override;
};

// ============ Tanh ============
class Tanh : public Module<float> {
public:
    Tanh() = default; ~Tanh() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
    void CleanGard() override;
};

// ============ SoftMax ============
class SoftMax : public Module<float> {
public:
    SoftMax() = default; ~SoftMax() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
    void CleanGard() override;
};

// ============ ELU ============
class ELU : public Module<float> {
    float alpha;
public:
    ELU(float a = 1.0f); ~ELU() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
    void CleanGard() override;
};

// ============ SELU ============
class SELU : public Module<float> {
public:
    SELU() = default; ~SELU() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
    void CleanGard() override;
};

// ============ Softplus ============
class Softplus : public Module<float> {
public:
    Softplus() = default; ~Softplus() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
    void CleanGard() override;
};

// ============ Mish ============
class Mish : public Module<float> {
public:
    Mish() = default; ~Mish() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
    void CleanGard() override;
};

// ============ GELU ============
class GELU : public Module<float> {
public:
    GELU() = default; ~GELU() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
    void CleanGard() override;
};

// ============ Swish ============
class Swish : public Module<float> {
public:
    Swish() = default; ~Swish() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
    void CleanGard() override;
};

} // namespace Activation
} // namespace NPCore
#endif
