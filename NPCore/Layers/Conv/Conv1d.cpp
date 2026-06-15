#include "Layers/Conv/Conv1d.h"

namespace NPCore {

Conv1d::Conv1d(int in_channels, int out_channels, int kernel_size,
               int stride, int padding,
               InitMode mode, bool use_bias,
               double mu, double sigma)
    : in_channels(in_channels), out_channels(out_channels),
      kernel_size(kernel_size), stride(stride), padding(padding),
      use_bias(use_bias)
{
    weight = new Matrix<float>(out_channels, in_channels * kernel_size);
    bias = use_bias ? new Matrix<float>(out_channels, 1) : nullptr;
    InitMatrixFunc(*weight, mode, {.mu = mu, .sigma = sigma});
    if (bias) InitMatrixFunc(*bias, InitMode::Zeros);
}

Conv1d::~Conv1d() {
    delete weight;
    delete bias;
}

// im2col for 1D: extract (kernel_size, in_channels) patches into columns
// Returns (in_channels * kernel_size, L_out) matrix
static Matrix<float> im2col_1d(const Matrix<float>& input,
                                int kernel_size, int stride, int padding) {
    int L = input.row;       // seq_len
    int C = input.col;       // in_channels
    int L_out = (L - kernel_size + 2 * padding) / stride + 1;

    int patch_size = C * kernel_size;
    Matrix<float> col(patch_size, L_out);

    for (int c = 0; c < C; ++c) {
        for (int k = 0; k < kernel_size; ++k) {
            int row_idx = c * kernel_size + k;
            for (int t = 0; t < L_out; ++t) {
                int src = t * stride + k - padding;
                col.at(row_idx, t) = (src >= 0 && src < L) ? input.at(src, c) : 0.0f;
            }
        }
    }
    return col;
}

// col2im for 1D: scatter column gradients back to input shape
static Matrix<float> col2im_1d(const Matrix<float>& col, int L, int C,
                                int kernel_size, int stride, int padding) {
    int L_out = col.col;
    Matrix<float> grad(L, C);

    for (int c = 0; c < C; ++c) {
        for (int k = 0; k < kernel_size; ++k) {
            int row_idx = c * kernel_size + k;
            for (int t = 0; t < L_out; ++t) {
                int dst = t * stride + k - padding;
                if (dst >= 0 && dst < L)
                    grad.at(dst, c) += col.at(row_idx, t);
            }
        }
    }
    return grad;
}

Matrix<float> Conv1d::forward(Matrix<float>& input) {
    NPCORE_ASSERT(input.col == in_channels,
                "Conv1d input channels mismatch. Expected %d, got %d",
                in_channels, input.col);

    int L_out = (input.row - kernel_size + 2 * padding) / stride + 1;
    Matrix<float> col = im2col_1d(input, kernel_size, stride, padding);
    Matrix<float> result_2d = (*weight) * col;  // (out_channels, L_out)

    auto* out = new Matrix<float>(L_out, out_channels);
    for (int oc = 0; oc < out_channels; ++oc) {
        float b = bias ? bias->at(oc, 0) : 0.0f;
        for (int t = 0; t < L_out; ++t)
            out->at(t, oc) = result_2d.at(oc, t) + b;
    }

    if (train_mode) {
        gard.push_back(new Matrix<float>(input));
        // Store col for backward (as gard element)
        gard.push_back(new Matrix<float>(col));
        output.push_back(out);
    } else {
        output.push_back(out);
    }

    return *out;
}

Matrix<float> Conv1d::backward(Matrix<float>& grad_output) {
    NPCORE_ASSERT(gard.size() >= 2, "Conv1d::backward: no saved context (train mode required)");

    int L_out = grad_output.row;
    int C_out = out_channels;
    int C_in = in_channels;
    int L_in = gard[gard.size()-2]->row;  // input row count
    (void)L_in;

    const Matrix<float>& col = gard.back() ? *gard[gard.size()-1] : *gard[gard.size()-2];
    Matrix<float>& input_cache = *gard[gard.size()-2];

    // dW: G2 * col^T
    // G2: (C_out, L_out)
    Matrix<float> G2(C_out, L_out);
    for (int c = 0; c < C_out; ++c)
        for (int t = 0; t < L_out; ++t)
            G2.at(c, t) = grad_output.at(t, c);

    Matrix<float> col_T = col.Translate();
    weight_grad_ = new Matrix<float>(G2 * col_T);

    if (bias) {
        bias_grad_ = new Matrix<float>(C_out, 1);
        for (int c = 0; c < C_out; ++c) {
            float sum = 0;
            for (int t = 0; t < L_out; ++t) sum += grad_output.at(t, c);
            bias_grad_->at(c, 0) = sum;
        }
    } else {
        bias_grad_ = nullptr;
    }

    // dX: W^T * G2, then col2im
    Matrix<float> W_T = weight->Translate();
    Matrix<float> dX_col = W_T * G2;

    return col2im_1d(dX_col, input_cache.row, C_in,
                     kernel_size, stride, padding);
}

std::vector<Matrix<float>*> Conv1d::getParams() {
    if (bias) return {weight, bias};
    return {weight};
}
Matrix<float>* Conv1d::getGard() { return gard.empty() ? nullptr : gard.back(); }
Matrix<float>* Conv1d::getOutput() { return output.empty() ? nullptr : output.back(); }

void Conv1d::CleanGard() {
    for (auto p : gard) delete p;
    gard.clear();
    for (auto p : output) delete p;
    output.clear();
    delete weight_grad_; weight_grad_ = nullptr;
    delete bias_grad_; bias_grad_ = nullptr;
}

} // namespace NPCore
