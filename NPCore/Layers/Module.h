#ifndef NPCORE_LAYERS_MODULE_H
#define NPCORE_LAYERS_MODULE_H

#include <vector>
#include "Core/Matrix.hpp"

namespace NPCore {

// =================================[Module Abstract Base Class]================================
// Base class for all neural network layers, virtual-base polymorphic design

template<typename T>
class Module {
protected:
    std::vector<Matrix<T>*> gard;
    std::vector<Matrix<T>*> output;

    // Pre-computed weight/bias gradients (filled by backward(), read by optimizer)
    Matrix<T>* weight_grad_ = nullptr;
    Matrix<T>* bias_grad_ = nullptr;

public:
    // Momentum (similar role to gradients)
    Matrix<T> m, v;

    // Train/eval mode
    bool train_mode = true;
    virtual void eval();
    virtual void train();

    // GPU: move all parameters to GPU / back to CPU
    virtual void cuda();
    virtual void cpu();

    Module() = default;
    virtual ~Module();

    // Forward pass
    virtual Matrix<T> forward(Matrix<T> &input) = 0;

    // Backward: given loss gradient w.r.t. output, return loss gradient w.r.t. input
    // Also stores weight/bias gradients in weight_grad_ / bias_grad_
    virtual Matrix<T> backward(Matrix<T>& grad_output) = 0;

    // Get parameters
    virtual std::vector<Matrix<float> *> getParams() = 0;

    // Get all gradients (corresponds to getParams order; RNN/LSTM have multiple weight matrices)
    virtual std::vector<Matrix<float>*> getAllGrads();

    // Get stored gradient (activation layers return derivative, parameter layers return input)
    virtual Matrix<T>* getGard() = 0;

    // Get output
    virtual Matrix<T>* getOutput() = 0;

    // Get pre-computed weight gradient (valid after backward)
    Matrix<T>* getWeightGrad();

    // Get pre-computed bias gradient (valid after backward)
    Matrix<T>* getBiasGrad();

    // Clear gradients
    virtual void CleanGard() = 0;
};

} // namespace NPCore

#endif // NPCORE_LAYERS_MODULE_H
