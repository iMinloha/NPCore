#ifndef COREPP_ACTIVATIONS_H
#define COREPP_ACTIVATIONS_H

#include <vector>
#include <cmath>
#include "Layers/Module.h"

namespace CoreNNSpace {
namespace Activation {

// ============ ReLU ============
class ReLU : public Module<float> {
public:
    ReLU() = default; ~ReLU() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return gard.empty()?nullptr:gard.back(); }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override { for(auto p:gard)delete p;gard.clear(); for(auto p:output)delete p;output.clear(); }
};

// ============ LeakyReLU ============
class LeakyReLU : public Module<float> {
    float alpha;
public:
    LeakyReLU(float a=0.01f):alpha(a){} ~LeakyReLU() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return gard.empty()?nullptr:gard.back(); }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override { for(auto p:gard)delete p;gard.clear(); for(auto p:output)delete p;output.clear(); }
};

// ============ Sigmoid ============
class Sigmoid : public Module<float> {
public:
    Sigmoid() = default; ~Sigmoid() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return gard.empty()?nullptr:gard.back(); }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override { for(auto p:gard)delete p;gard.clear(); for(auto p:output)delete p;output.clear(); }
};

// ============ Tanh ============
class Tanh : public Module<float> {
public:
    Tanh() = default; ~Tanh() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return gard.empty()?nullptr:gard.back(); }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override { for(auto p:gard)delete p;gard.clear(); for(auto p:output)delete p;output.clear(); }
};

// ============ SoftMax ============
class SoftMax : public Module<float> {
public:
    SoftMax() = default; ~SoftMax() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return gard.empty()?nullptr:gard.back(); }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override { for(auto p:gard)delete p;gard.clear(); for(auto p:output)delete p;output.clear(); }
};

// ============ ELU ============
class ELU : public Module<float> {
    float alpha;
public:
    ELU(float a=1.0f):alpha(a){} ~ELU() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return gard.empty()?nullptr:gard.back(); }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override { for(auto p:gard)delete p;gard.clear(); for(auto p:output)delete p;output.clear(); }
};

// ============ SELU ============
class SELU : public Module<float> {
public:
    SELU() = default; ~SELU() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return gard.empty()?nullptr:gard.back(); }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override { for(auto p:gard)delete p;gard.clear(); for(auto p:output)delete p;output.clear(); }
};

// ============ Softplus ============
class Softplus : public Module<float> {
public:
    Softplus() = default; ~Softplus() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return gard.empty()?nullptr:gard.back(); }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override { for(auto p:gard)delete p;gard.clear(); for(auto p:output)delete p;output.clear(); }
};

// ============ Mish ============
class Mish : public Module<float> {
public:
    Mish() = default; ~Mish() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return gard.empty()?nullptr:gard.back(); }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override { for(auto p:gard)delete p;gard.clear(); for(auto p:output)delete p;output.clear(); }
};

// ============ GELU ============
class GELU : public Module<float> {
public:
    GELU() = default; ~GELU() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return gard.empty()?nullptr:gard.back(); }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override { for(auto p:gard)delete p;gard.clear(); for(auto p:output)delete p;output.clear(); }
};

// ============ Swish ============
class Swish : public Module<float> {
public:
    Swish() = default; ~Swish() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return gard.empty()?nullptr:gard.back(); }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override { for(auto p:gard)delete p;gard.clear(); for(auto p:output)delete p;output.clear(); }
};

} // namespace Activation
} // namespace CoreNNSpace
#endif
