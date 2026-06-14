#ifndef NPCORE_LAYERS_CONVTRANSPOSE2D_H
#define NPCORE_LAYERS_CONVTRANSPOSE2D_H

#include "Layers/Module.h"
#include "Layers/ParamInit.h"
#include "Core/ConvUtils.h"

namespace NPCore {

// =================================[ConvTranspose2d - Transposed Convolution]================================
// Also known as "deconvolution". Essential for U-Net decoders and GAN generators.
// Input: (H_in, W_in, C_in), Output: (H_out, W_out, C_out)
// H_out = (H_in-1)*stride - 2*padding + kernel_size

class ConvTranspose2d : public Module<float> {
private:
    Matrix<float>* weight;
    Matrix<float>* bias;
    Matrix<float>* col_cache = nullptr;
    int in_channels, out_channels, kernel_size, stride, padding;

    mutable Matrix<float>* weight_2d_T_cache_ = nullptr;
    mutable bool weight_2d_T_dirty_ = true;

public:
    ConvTranspose2d(int in_channels, int out_channels, int kernel_size = 3,
                    int stride = 2, int padding = 1,
                    InitMode mode = InitMode::XavierUniform,
                    double mu = 0.0, double sigma = 1.0);
    ~ConvTranspose2d() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;

    // Lazy-cached transposed weight layout for GEMM
    Matrix<float> weight_2d_T() const;
};

} // namespace NPCore

#endif
