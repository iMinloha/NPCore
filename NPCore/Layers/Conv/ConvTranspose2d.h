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

    std::vector<Matrix<float>*> getParams() override {
        weight_2d_T_dirty_ = true;
        return {weight, bias};
    }
    Matrix<float>* getGard() override { return gard.empty() ? nullptr : gard.back(); }
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }

    // Lazy-cached transposed weight layout for GEMM (kept inline for performance)
    Matrix<float> weight_2d_T() const {
        if (weight_2d_T_cache_ && !weight_2d_T_dirty_) return *weight_2d_T_cache_;
        delete weight_2d_T_cache_;
        int K = kernel_size, C_in = in_channels, C_out = out_channels, K2 = K * K;
        auto* W = new Matrix<float>(C_out * K2, C_in);
        for (int in_ch = 0; in_ch < C_in; ++in_ch) {
            for (int out_ch = 0; out_ch < C_out; ++out_ch) {
                int ch_offset = in_ch * C_out + out_ch;
                int base_row = out_ch * K2;
                for (int ki = 0; ki < K; ++ki)
                    for (int kj = 0; kj < K; ++kj)
                        W->at(base_row + ki * K + kj, in_ch) = weight->at(ki, kj, ch_offset);
            }
        }
        weight_2d_T_cache_ = W;
        weight_2d_T_dirty_ = false;
        return *W;
    }
};

} // namespace NPCore
#endif
