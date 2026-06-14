#include "Layers/Normalization/GroupNorm.h"
#include <cmath>

namespace NPCore {

GroupNorm::GroupNorm(int num_groups, int channels)
    : num_groups(num_groups), channels(channels),
      channels_per_group(channels / num_groups)
{
    NPCORE_ASSERT(channels % num_groups == 0,
                "channels (%d) must be divisible by num_groups (%d)", channels, num_groups);

    gamma = new Matrix<float>(channels, 1);
    beta  = new Matrix<float>(channels, 1);
    InitMatrixFunc(*gamma, Ones);
    InitMatrixFunc(*beta,  Zeros);
}

GroupNorm::~GroupNorm() {
    delete gamma; delete beta;
    delete dgamma; delete dbeta;
}

std::vector<Matrix<float>*> GroupNorm::getParams() { return {gamma, beta}; }
std::vector<Matrix<float>*> GroupNorm::getAllGrads() { return {dgamma, dbeta}; }
Matrix<float>* GroupNorm::getGard() { return nullptr; }
Matrix<float>* GroupNorm::getOutput() { return output.empty() ? nullptr : output.back(); }

Matrix<float> GroupNorm::forward(Matrix<float>& input) {
    int H = input.row, W = input.col, C = input.channel;
    int spatial = H * W;
    int cpg = channels_per_group;

    auto* out = new Matrix<float>(H, W, C);

    if (train_mode) gard.push_back(new Matrix<float>(input));

    for (int g = 0; g < num_groups; ++g) {
        int ch_start = g * cpg;
        int ch_end   = ch_start + cpg;
        int N = spatial * cpg;

        float mean = 0.0f;
        for (int c = ch_start; c < ch_end; ++c)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j)
                    mean += input.at(i, j, c);
        mean /= N;

        float var = 0.0f;
        for (int c = ch_start; c < ch_end; ++c)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j) {
                    float d = input.at(i, j, c) - mean;
                    var += d * d;
                }
        var /= N;

        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int c = ch_start; c < ch_end; ++c) {
            float gm = gamma->at(c, 0), bt = beta->at(c, 0);
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j)
                    out->at(i, j, c) = gm * (input.at(i, j, c) - mean) * inv_std + bt;
        }
    }

    output.push_back(out);
    return *out;
}

Matrix<float> GroupNorm::backward(Matrix<float>& grad_output) {
    int H = grad_output.row, W = grad_output.col, C = grad_output.channel;
    int spatial = H * W;
    int cpg = channels_per_group;

    auto& x = *gard.back();

    dgamma = new Matrix<float>(C, 1);
    dbeta  = new Matrix<float>(C, 1);
    auto* gi = new Matrix<float>(H, W, C);

    for (int g = 0; g < num_groups; ++g) {
        int ch_start = g * cpg;
        int ch_end   = ch_start + cpg;
        int N = spatial * cpg;

        float mean = 0.0f;
        for (int c = ch_start; c < ch_end; ++c)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j)
                    mean += x.at(i, j, c);
        mean /= N;

        float var = 0.0f;
        for (int c = ch_start; c < ch_end; ++c)
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j) {
                    float d = x.at(i, j, c) - mean;
                    var += d * d;
                }
        var /= N;

        float inv_std = 1.0f / std::sqrt(var + eps);

        for (int c = ch_start; c < ch_end; ++c) {
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    dgamma->at(c, 0) += grad_output.at(i, j, c)
                        * (x.at(i, j, c) - mean) * inv_std;
                    dbeta->at(c, 0) += grad_output.at(i, j, c);
                }
            }
        }

        for (int c = ch_start; c < ch_end; ++c) {
            float gamma_c = gamma->at(c, 0);

            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    float x_hat = (x.at(i, j, c) - mean) * inv_std;
                    float dy_hat = grad_output.at(i, j, c) * gamma_c;

                    float sum_dy_hat = 0.0f, sum_dy_hat_x_hat = 0.0f;
                    for (int cc = ch_start; cc < ch_end; ++cc) {
                        for (int ii = 0; ii < H; ++ii) {
                            for (int jj = 0; jj < W; ++jj) {
                                float dy_h = grad_output.at(ii, jj, cc) * gamma->at(cc, 0);
                                sum_dy_hat += dy_h;
                                sum_dy_hat_x_hat += dy_h
                                    * (x.at(ii, jj, cc) - mean) * inv_std;
                            }
                        }
                    }

                    gi->at(i, j, c) = inv_std / N
                        * (N * dy_hat - sum_dy_hat - x_hat * sum_dy_hat_x_hat);
                }
            }
        }
    }

    return *gi;
}

void GroupNorm::CleanGard() {
    for (auto p : gard) { delete p; }
    gard.clear();
    for (auto p : output) { delete p; }
    output.clear();
    delete dgamma; dgamma = nullptr;
    delete dbeta;  dbeta  = nullptr;
}

} // namespace NPCore
