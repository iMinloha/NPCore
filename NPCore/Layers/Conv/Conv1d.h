#ifndef NPCORE_LAYERS_CONV1D_H
#define NPCORE_LAYERS_CONV1D_H

#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace NPCore {

// =================================[Conv1d - 1D Convolution]================================
// Input:  (seq_len, in_channels)  — each row = one time step
// Output: (seq_len_out, out_channels)
// seq_len_out = (seq_len - kernel_size + 2*padding) / stride + 1

class NPCORE_API Conv1d : public Module<float> {
    Matrix<float>* weight;  // (out_channels, in_channels * kernel_size)
    Matrix<float>* bias;
    bool use_bias;
    int in_channels, out_channels, kernel_size, stride, padding;

public:
    Conv1d(int in_channels, int out_channels, int kernel_size = 3,
           int stride = 1, int padding = 0,
           InitMode mode = InitMode::XavierUniform,
           bool use_bias = true,
           double mu = 0.0, double sigma = 1.0);
    ~Conv1d() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
};

} // namespace NPCore

#endif
