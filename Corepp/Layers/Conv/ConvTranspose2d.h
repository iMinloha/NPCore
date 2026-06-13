#ifndef COREPP_LAYERS_CONVTRANSPOSE2D_H
#define COREPP_LAYERS_CONVTRANSPOSE2D_H

#include <vector>
#include <cmath>
#include "Layers/Module.h"
#include "Layers/ParamInit.h"
#include "Core/ConvUtils.h"

namespace CoreNNSpace {

// =================================[ConvTranspose2d — 转置卷积 / 反卷积]================================
// Reference: Dumoulin & Visin (2016) "A guide to convolution arithmetic for deep learning"
//
// Also known as "deconvolution" or "fractionally-strided convolution".
// Essential for U-Net decoders and GAN generators where upsampling is needed.
//
// Input:  (H_in, W_in, C_in)
// Output: (H_out, W_out, C_out)
//   H_out = (H_in - 1) * stride - 2 * padding + kernel_size
//   W_out = (W_in - 1) * stride - 2 * padding + kernel_size
//
// The forward pass is equivalent to the backward pass of Conv2d:
//   1. Multiply transposed weight matrix with flattened input  →  col
//   2. col2im to scatter columns back to spatial output
//
// Weight shape: (kernel_size, kernel_size, C_in * C_out)
//   — same convention as Conv2d (kernel_h, kernel_w, in_ch*out_ch)

class ConvTranspose2d : public Module<float> {
private:
    Matrix<float>* weight;
    Matrix<float>* bias;
    Matrix<float>* col_cache = nullptr;

    int in_channels, out_channels, kernel_size, stride, padding;

    // Cached weight_2d_T: (C_out * K * K, C_in) — transposed layout for matmul
    mutable Matrix<float>* weight_2d_T_cache_ = nullptr;
    mutable bool weight_2d_T_dirty_ = true;

public:
    ConvTranspose2d(int in_channels, int out_channels, int kernel_size = 3,
                    int stride = 2, int padding = 1,
                    InitMode mode = InitMode::XavierUniform,
                    double mu = 0.0, double sigma = 1.0)
        : in_channels(in_channels), out_channels(out_channels),
          kernel_size(kernel_size), stride(stride), padding(padding)
    {
        weight = new Matrix<float>(kernel_size, kernel_size, in_channels * out_channels);
        bias   = new Matrix<float>(out_channels, 1);

        InitMatrixFunc(*weight, mode, {.mu = mu, .sigma = sigma});
        InitMatrixFunc(*bias, InitMode::Zeros);
    }

    ~ConvTranspose2d() override {
        delete weight; delete bias; delete col_cache;
        delete weight_2d_T_cache_;
    }

    // ===============================[weight_2d_T — Transposed Weight Layout]===============================
    // Returns (C_out * K * K, C_in) — the transposed 2D weight for GEMM.
    // This is the transpose of what Conv2d.weight_2d() returns.
    Matrix<float> weight_2d_T() const {
        if (weight_2d_T_cache_ && !weight_2d_T_dirty_) return *weight_2d_T_cache_;

        delete weight_2d_T_cache_;
        int K = kernel_size;
        int C_in = in_channels, C_out = out_channels;
        int K2 = K * K;

        auto* W = new Matrix<float>(C_out * K2, C_in);

        // weight is (K, K, C_in * C_out) with channel offset = in_ch * C_out + out_ch
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

    // ===============================[Forward]===============================
    // 1. Flatten input: (C_in, H_in*W_in)
    // 2. col = weight_2d_T * input_2d: (C_out*K*K, H_in*W_in)
    // 3. col2im → (H_out, W_out, C_out) + bias
    Matrix<float> forward(Matrix<float>& input) override {
        COREPP_ASSERT(input.channel == in_channels,
                    "Input channels mismatch. Expected %d, got %d", in_channels, input.channel);

        int H_in = input.row, W_in = input.col;
        int K = kernel_size;

        // Output spatial dims (transposed convolution formula)
        int H_out = (H_in - 1) * stride - 2 * padding + K;
        int W_out = (W_in - 1) * stride - 2 * padding + K;

        // Step 1: Flatten input → (C_in, H_in * W_in)
        Matrix<float> input_2d(in_channels, H_in * W_in);
        for (int c = 0; c < in_channels; ++c)
            for (int i = 0; i < H_in; ++i)
                for (int j = 0; j < W_in; ++j)
                    input_2d.at(c, i * W_in + j) = input.at(i, j, c);

        // Step 2: col = W_T * input_2d  →  (C_out * K * K, H_in * W_in)
        const Matrix<float>& W_T = weight_2d_T();
        Matrix<float> col = W_T * input_2d;

        // Step 3: col2im → (H_out, W_out, C_out)
        // col has rows = C_out * K * K, cols = H_in * W_in
        // For ConvTranspose2d, each input position maps to a K×K patch in the output.
        // Columns of 'col' correspond to input positions; each column is a K×K×C_out patch.
        auto* out = new Matrix<float>(H_out, W_out, out_channels);

        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ki = 0; ki < K; ++ki) {
                for (int kj = 0; kj < K; ++kj) {
                    int row_idx = oc * K * K + ki * K + kj;

                    for (int ih = 0; ih < H_in; ++ih) {
                        for (int iw = 0; iw < W_in; ++iw) {
                            int col_idx = ih * W_in + iw;

                            int oh = ih * stride + ki - padding;
                            int ow = iw * stride + kj - padding;

                            if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out)
                                out->at(oh, ow, oc) += col.at(row_idx, col_idx);
                        }
                    }
                }
            }
        }

        // Add bias
        for (int oc = 0; oc < out_channels; ++oc)
            for (int i = 0; i < H_out; ++i)
                for (int j = 0; j < W_out; ++j)
                    out->at(i, j, oc) += bias->at(oc, 0);

        if (train_mode) {
            gard.push_back(new Matrix<float>(input));
            delete col_cache;
            col_cache = new Matrix<float>(col);
            output.push_back(out);
        } else {
            output.push_back(out);
        }

        return *out;
    }

    // ===============================[Backward]===============================
    // The backward of ConvTranspose2d is equivalent to the forward of a regular Conv2d
    // with the same kernel/stride/padding but swapping in/out channels.
    Matrix<float> backward(Matrix<float>& grad_output) override {
        COREPP_ASSERT(col_cache != nullptr,
                    "ConvTranspose2d::backward called before forward (train mode required)");

        int K = kernel_size;
        int C_in = in_channels, C_out = out_channels;
        int H_out = grad_output.row, W_out = grad_output.col;

        Matrix<float>& input_cache = *gard.back();
        int H_in = input_cache.row, W_in = input_cache.col;

        // --- Weight gradient ---
        // dW = im2col(grad_output) * input_2d^T, then reshape to (K, K, C_in*C_out)
        // im2col on grad_output: (H_out, W_out, C_out) → (C_out * K * K, H_in * W_in)
        Matrix<float> go_col = im2col(grad_output, K, K, stride, padding);

        // input_2d: (C_in, H_in * W_in)
        Matrix<float> input_2d(C_in, H_in * W_in);
        for (int c = 0; c < C_in; ++c)
            for (int i = 0; i < H_in; ++i)
                for (int j = 0; j < W_in; ++j)
                    input_2d.at(c, i * W_in + j) = input_cache.at(i, j, c);

        Matrix<float> input_2d_T = input_2d.Translate();  // (H_in*W_in, C_in)
        Matrix<float> dW_2d = go_col * input_2d_T;  // (C_out*K*K, H_in*W_in) * (H_in*W_in, C_in) = (C_out*K*K, C_in)

        // Reshape to (K, K, C_in * C_out)
        weight_grad_ = new Matrix<float>(K, K, C_in * C_out);
        int K2 = K * K;
        for (int in_ch = 0; in_ch < C_in; ++in_ch) {
            for (int out_ch = 0; out_ch < C_out; ++out_ch) {
                int ch_offset = in_ch * C_out + out_ch;
                int base_row = out_ch * K2;
                for (int ki = 0; ki < K; ++ki)
                    for (int kj = 0; kj < K; ++kj)
                        weight_grad_->at(ki, kj, ch_offset) = dW_2d.at(base_row + ki * K + kj, in_ch);
            }
        }

        // --- Bias gradient ---
        bias_grad_ = new Matrix<float>(C_out, 1);
        for (int c = 0; c < C_out; ++c) {
            float sum = 0.0f;
            for (int i = 0; i < H_out; ++i)
                for (int j = 0; j < W_out; ++j)
                    sum += grad_output.at(i, j, c);
            bias_grad_->at(c, 0) = sum;
        }

        // --- Input gradient ---
        // d_input = W * im2col(grad_output) via col2im
        // W (as forward) is (C_out*K*K, C_in), so W^T is (C_in, C_out*K*K)
        // d_col = W^T * go_col ... actually:
        // d_input is the gradient w.r.t. the input.
        // In forward: col = W_T * input_2d
        // So d_input_2d = W_T^T * d_col = W * d_col
        // But d_col = go_col (from the chain rule)
        //
        // Actually: forward is col = W^T * input_2d
        // dL/d(input_2d) = W * dL/d(col)
        // dL/d(col) = go_col (since col→out is just col2im, and col2im's backward is im2col)
        //
        // Wait — let me think more carefully. The forward chain is:
        // input → input_2d → col → out
        //
        // col = W_T * input_2d
        // out = col2im(col) + bias
        //
        // For backward:
        // d_out = grad_output
        // d_col = im2col(d_out)  (because col2im's backward is im2col)
        // d_input_2d = W_T^T * d_col
        // d_input = reshape(d_input_2d)
        //
        // So d_col = go_col = im2col(grad_output)
        // And d_input_2d = W * go_col  (since W is (C_out*K*K, C_in) and W_T is (C_out*K*K, C_in)... wait, W_T = weight_2d_T which is (C_out*K*K, C_in))

        // W is weight_2d_T: (C_out*K*K, C_in)
        // d_input_2d = W^T * go_col = (C_in, C_out*K*K) * (C_out*K*K, H_in*W_in) = (C_in, H_in*W_in)
        const Matrix<float>& W = weight_2d_T();
        Matrix<float> W_T_mat = W.Translate();  // (C_in, C_out*K*K)
        Matrix<float> d_input_2d = W_T_mat * go_col;  // (C_in, H_in*W_in)

        // Reshape back to (H_in, W_in, C_in)
        auto* d_input = new Matrix<float>(H_in, W_in, C_in);
        for (int c = 0; c < C_in; ++c)
            for (int i = 0; i < H_in; ++i)
                for (int j = 0; j < W_in; ++j)
                    d_input->at(i, j, c) = d_input_2d.at(c, i * W_in + j);

        return *d_input;
    }

    // ===============================[Parameter Access]===============================
    std::vector<Matrix<float>*> getParams() override {
        weight_2d_T_dirty_ = true;
        return {weight, bias};
    }
    Matrix<float>* getGard() override { return gard.empty() ? nullptr : gard.back(); }
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }

    void CleanGard() override {
        for (auto p : gard)   delete p;
        for (auto p : output) delete p;
        gard.clear();
        output.clear();
        delete weight_grad_; weight_grad_ = nullptr;
        delete bias_grad_;  bias_grad_  = nullptr;
    }
};

} // namespace CoreNNSpace

#endif // COREPP_LAYERS_CONVTRANSPOSE2D_H
