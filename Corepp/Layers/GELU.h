#ifndef COREPP_LAYERS_GELU_H
#define COREPP_LAYERS_GELU_H
#include <cmath>
#include "Layers/Activation.h"
namespace CoreNNSpace {
namespace Activation {

// =================================[GELU — Gaussian Error Linear Unit]================================
// gelu(x) = 0.5 * x * (1 + erf(x/√2))
// Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715*x³)))
class GELU : public Module<float> {
    static constexpr float A = 0.7978845608f;  // √(2/π)
    static constexpr float B = 0.044715f;
public:
    GELU() = default; ~GELU() override = default;

    Matrix<float> forward(Matrix<float>& input) override {
        auto* r = new Matrix<float>(input.row, input.col, input.channel);
        auto* d = new Matrix<float>(input.row, input.col, input.channel);
        int n = input.row * input.col * input.channel;
        for (int i = 0; i < n; ++i) {
            float x = input.data_ptr()[i];
            float x3 = x * x * x;
            float t = std::tanh(A * (x + B * x3));
            r->data_ptr()[i] = 0.5f * x * (1.0f + t);
            // Derivative (approximate)
            float sech2 = 1.0f - t * t;
            d->data_ptr()[i] = 0.5f * (1.0f + t) + 0.5f * x * A * (1.0f + 3.0f * B * x * x) * sech2;
        }
        gard.push_back(d); output.push_back(r); return *r;
    }

    Matrix<float> backward(Matrix<float>& grad_output) override {
        if (gard.empty()) return grad_output;
        return grad_output & (*gard.back());
    }

    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return gard.empty()?nullptr:gard.back(); }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override { for(auto p:gard)delete p; gard.clear(); for(auto p:output)delete p; output.clear(); }
};

// =================================[Swish / SiLU]================================
// swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
class Swish : public Module<float> {
public:
    Swish() = default; ~Swish() override = default;

    Matrix<float> forward(Matrix<float>& input) override {
        auto* r = new Matrix<float>(input.row, input.col, input.channel);
        auto* d = new Matrix<float>(input.row, input.col, input.channel);
        int n = input.row * input.col * input.channel;
        for (int i = 0; i < n; ++i) {
            float x = input.data_ptr()[i];
            float s = 1.0f / (1.0f + std::exp(-x));
            r->data_ptr()[i] = x * s;
            d->data_ptr()[i] = s * (1.0f + x * (1.0f - s));
        }
        gard.push_back(d); output.push_back(r); return *r;
    }

    Matrix<float> backward(Matrix<float>& grad_output) override {
        if (gard.empty()) return grad_output;
        return grad_output & (*gard.back());
    }

    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return gard.empty()?nullptr:gard.back(); }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override { for(auto p:gard)delete p; gard.clear(); for(auto p:output)delete p; output.clear(); }
};

} // namespace Activation
} // namespace CoreNNSpace
#endif
