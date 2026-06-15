#include "Layers/Conv/ConvTranspose2d.h"
#include <cmath>

namespace NPCore {

ConvTranspose2d::ConvTranspose2d(int in_channels, int out_channels, int kernel_size,
                                 int stride, int padding,
                                 InitMode mode, bool use_bias,
                                 double mu, double sigma)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_size(kernel_size), stride(stride), padding(padding), use_bias(use_bias)
{
    weight = new Matrix<float>(kernel_size, kernel_size, in_channels * out_channels);
    bias   = use_bias ? new Matrix<float>(out_channels, 1) : nullptr;
    InitMatrixFunc(*weight, mode, {.mu = mu, .sigma = sigma});
    if (bias) InitMatrixFunc(*bias, InitMode::Zeros);
}

ConvTranspose2d::~ConvTranspose2d() {
    delete weight; delete bias; delete col_cache;
    delete weight_2d_T_cache_;
}

Matrix<float> ConvTranspose2d::forward(Matrix<float>& input) {
    NPCORE_ASSERT(input.channel == in_channels,
                "Input channels mismatch. Expected %d, got %d", in_channels, input.channel);

    int H_in = input.row, W_in = input.col;
    int K = kernel_size;
    int H_out = (H_in - 1) * stride - 2 * padding + K;
    int W_out = (W_in - 1) * stride - 2 * padding + K;

    Matrix<float> input_2d(in_channels, H_in * W_in);
    for (int c = 0; c < in_channels; ++c)
        for (int i = 0; i < H_in; ++i)
            for (int j = 0; j < W_in; ++j)
                input_2d.at(c, i * W_in + j) = input.at(i, j, c);

    const Matrix<float>& W_T = weight_2d_T();
    Matrix<float> col = W_T * input_2d;

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

    for (int oc = 0; oc < out_channels; ++oc)
        for (int i = 0; i < H_out; ++i)
            for (int j = 0; j < W_out; ++j)
                out->at(i, j, oc) += bias ? bias->at(oc, 0) : 0.0f;

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

Matrix<float> ConvTranspose2d::backward(Matrix<float>& grad_output) {
    NPCORE_ASSERT(col_cache != nullptr,
                "ConvTranspose2d::backward called before forward (train mode required)");

    int K = kernel_size;
    int C_in = in_channels, C_out = out_channels;
    int H_out = grad_output.row, W_out = grad_output.col;

    Matrix<float>& input_cache = *gard.back();
    int H_in = input_cache.row, W_in = input_cache.col;

    Matrix<float> go_col = im2col(grad_output, K, K, stride, padding);

    Matrix<float> input_2d(C_in, H_in * W_in);
    for (int c = 0; c < C_in; ++c)
        for (int i = 0; i < H_in; ++i)
            for (int j = 0; j < W_in; ++j)
                input_2d.at(c, i * W_in + j) = input_cache.at(i, j, c);

    Matrix<float> input_2d_T = input_2d.Translate();
    Matrix<float> dW_2d = go_col * input_2d_T;

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

    if (bias) {
        bias_grad_ = new Matrix<float>(C_out, 1);
        for (int c = 0; c < C_out; ++c) {
            float sum = 0.0f;
            for (int i = 0; i < H_out; ++i)
                for (int j = 0; j < W_out; ++j)
                    sum += grad_output.at(i, j, c);
            bias_grad_->at(c, 0) = sum;
        }
    } else {
        bias_grad_ = nullptr;
    }

    const Matrix<float>& W = weight_2d_T();
    Matrix<float> W_T_mat = W.Translate();
    Matrix<float> d_input_2d = W_T_mat * go_col;

    auto* d_input = new Matrix<float>(H_in, W_in, C_in);
    for (int c = 0; c < C_in; ++c)
        for (int i = 0; i < H_in; ++i)
            for (int j = 0; j < W_in; ++j)
                d_input->at(i, j, c) = d_input_2d.at(c, i * W_in + j);

    return *d_input;
}

std::vector<Matrix<float>*> ConvTranspose2d::getParams() {
    weight_2d_T_dirty_ = true;
    if (bias) return {weight, bias};
    return {weight};
}
Matrix<float>* ConvTranspose2d::getGard() { return gard.empty() ? nullptr : gard.back(); }
Matrix<float>* ConvTranspose2d::getOutput() { return output.empty() ? nullptr : output.back(); }

Matrix<float> ConvTranspose2d::weight_2d_T() const {
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

void ConvTranspose2d::CleanGard() {
    for (auto p : gard)   delete p;
    for (auto p : output) { delete p; }
    gard.clear();
    output.clear();
    delete weight_grad_; weight_grad_ = nullptr;
    delete bias_grad_;  bias_grad_  = nullptr;
}

} // namespace NPCore
