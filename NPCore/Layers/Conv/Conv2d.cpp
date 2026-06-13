#include "Layers/Conv/Conv2d.h"

namespace NPCore {

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
               InitMode mode, double mu, double sigma)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_size(kernel_size), stride(stride), padding(padding) {

    weight = new Matrix<float>(kernel_size, kernel_size, in_channels * out_channels);
    bias = new Matrix<float>(out_channels, 1);

    InitMatrixFunc(*weight, mode, {.mu = mu, .sigma = sigma});
    InitMatrixFunc(*bias, InitMode::Zeros);
}

// weight_2d with lazy caching
Matrix<float> Conv2d::weight_2d() const {
    if (weight_2d_cache_ && !weight_2d_dirty_) return *weight_2d_cache_;

    delete weight_2d_cache_;
    int K = kernel_size;
    auto* W = new Matrix<float>(out_channels, in_channels * K * K);
    for (int out_ch = 0; out_ch < out_channels; ++out_ch) {
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            int ch_offset = out_ch * in_channels + in_ch;
            int base_col = in_ch * K * K;
            for (int ki = 0; ki < K; ++ki)
                for (int kj = 0; kj < K; ++kj)
                    W->at(out_ch, base_col + ki * K + kj) = weight->at(ki, kj, ch_offset);
        }
    }
    weight_2d_cache_ = W;
    weight_2d_dirty_ = false;
    return *W;
}

Matrix<float> Conv2d::grad_output_2d(const Matrix<float>& grad_output) const {
    int H = grad_output.row, W = grad_output.col, C = grad_output.channel;
    Matrix<float> G(C, H * W);
    for (int c = 0; c < C; ++c)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                G.at(c, i * W + j) = grad_output.at(i, j, c);
    return G;
}

Matrix<float> Conv2d::forward(Matrix<float>& input) {
    NPCORE_ASSERT(input.channel == in_channels,
                "Input channels mismatch. Expected %d, got %d", in_channels, input.channel);

    int K = kernel_size;
    int H_out = (input.row - K + 2 * padding) / stride + 1;
    int W_out = (input.col - K + 2 * padding) / stride + 1;

    // im2col + GEMM (uses cached weight_2d)
    Matrix<float> col = im2col(input, K, K, stride, padding);
    const Matrix<float>& W2 = weight_2d();
    Matrix<float> result_2d = W2 * col;

    // Reshape + bias
    auto* out = new Matrix<float>(H_out, W_out, out_channels);
    for (int c = 0; c < out_channels; ++c)
        for (int i = 0; i < H_out; ++i)
            for (int j = 0; j < W_out; ++j)
                out->at(i, j, c) = result_2d.at(c, i * W_out + j) + bias->at(c, 0);

    // Inference: skip gradient storage for speed
    if (train_mode) {
        gard.push_back(new Matrix<float>(input));
        delete col_cache;
        col_cache = new Matrix<float>(col);
        output.push_back(out);
    } else {
        // Eval mode: no gradient needed
        output.push_back(out);
    }

    return *out;
}

Matrix<float> Conv2d::backward(Matrix<float>& grad_output) {
    NPCORE_ASSERT(col_cache != nullptr, "Conv2d::backward called before forward (train mode required)");

    int K = kernel_size, C_out = out_channels, C_in = in_channels;
    int H_out = grad_output.row, W_out = grad_output.col;

    const Matrix<float>& col = *col_cache;
    Matrix<float>& input_cache = *gard.back();

    // dW: G2 * col^T (uses cached weight_2d via weight_2d() for gradient computation)
    Matrix<float> G2 = grad_output_2d(grad_output);
    Matrix<float> col_T = col.Translate();
    Matrix<float> dW_2d = G2 * col_T;

    weight_grad_ = new Matrix<float>(K, K, C_in * C_out);
    for (int out_ch = 0; out_ch < C_out; ++out_ch) {
        for (int in_ch = 0; in_ch < C_in; ++in_ch) {
            int ch_offset = out_ch * C_in + in_ch;
            int base_col = in_ch * K * K;
            for (int ki = 0; ki < K; ++ki)
                for (int kj = 0; kj < K; ++kj)
                    weight_grad_->at(ki, kj, ch_offset) = dW_2d.at(out_ch, base_col + ki * K + kj);
        }
    }

    bias_grad_ = new Matrix<float>(C_out, 1);
    for (int c = 0; c < C_out; ++c) {
        float sum = 0.0f;
        for (int i = 0; i < H_out; ++i)
            for (int j = 0; j < W_out; ++j)
                sum += grad_output.at(i, j, c);
        bias_grad_->at(c, 0) = sum;
    }

    const Matrix<float>& W2 = weight_2d();
    Matrix<float> W2_T = W2.Translate();
    Matrix<float> dX_col = W2_T * G2;

    return col2im(dX_col, input_cache.row, input_cache.col,
                  C_in, K, K, stride, padding);
}

} // namespace NPCore
