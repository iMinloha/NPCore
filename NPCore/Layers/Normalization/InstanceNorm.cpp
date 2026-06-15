#include "Layers/Normalization/InstanceNorm.h"
#include <cmath>

namespace NPCore {

// =================================[InstanceNorm1d]================================

InstanceNorm1d::InstanceNorm1d(int features) : features(features) {
    gamma = new Matrix<float>(features, 1); InitMatrixFunc(*gamma, Ones);
    beta  = new Matrix<float>(features, 1); InitMatrixFunc(*beta,  Zeros);
}

InstanceNorm1d::~InstanceNorm1d() {
    delete gamma; delete beta;
    delete dgamma; delete dbeta;
}

Matrix<float> InstanceNorm1d::forward(Matrix<float>& input) {
    int N = input.row, C = input.col;
    auto* out = new Matrix<float>(N, C);

    // Per-sample: normalize each row independently
    for (int i = 0; i < N; ++i) {
        float mean = 0, var = 0;
        for (int f = 0; f < C; ++f) mean += input.at(i, f);
        mean /= C;
        for (int f = 0; f < C; ++f) { float d = input.at(i, f) - mean; var += d * d; }
        var /= C;
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int f = 0; f < C; ++f)
            out->at(i, f) = gamma->at(f, 0) * (input.at(i, f) - mean) * inv_std + beta->at(f, 0);
    }

    if (train_mode) gard.push_back(new Matrix<float>(input));
    output.push_back(out);
    return *out;
}

Matrix<float> InstanceNorm1d::backward(Matrix<float>& grad_output) {
    int N = grad_output.row, C = grad_output.col;
    auto& x = *gard.back();
    dgamma = new Matrix<float>(C, 1);
    dbeta  = new Matrix<float>(C, 1);
    auto* gi = new Matrix<float>(N, C);

    for (int i = 0; i < N; ++i) {
        float mean = 0, var = 0;
        for (int f = 0; f < C; ++f) mean += x.at(i, f);
        mean /= C;
        for (int f = 0; f < C; ++f) { float d = x.at(i, f) - mean; var += d * d; }
        var /= C;
        float inv_std = 1.0f / std::sqrt(var + eps);

        float dx_hat_sum = 0, dx_hat_x_sum = 0;
        for (int f = 0; f < C; ++f) {
            float dx_hat = grad_output.at(i, f) * gamma->at(f, 0);
            dx_hat_sum += dx_hat;
            dx_hat_x_sum += dx_hat * (x.at(i, f) - mean);
            dgamma->at(f, 0) += grad_output.at(i, f) * (x.at(i, f) - mean) * inv_std;
            dbeta->at(f, 0)  += grad_output.at(i, f);
        }
        for (int f = 0; f < C; ++f) {
            float dx_hat = grad_output.at(i, f) * gamma->at(f, 0);
            gi->at(i, f) = inv_std / C * (C * dx_hat - dx_hat_sum
                - (x.at(i, f) - mean) * inv_std * inv_std * dx_hat_x_sum);
        }
    }
    return *gi;
}

// =================================[InstanceNorm2d]================================

InstanceNorm2d::InstanceNorm2d(int channels) : channels(channels) {
    gamma = new Matrix<float>(channels, 1); InitMatrixFunc(*gamma, Ones);
    beta  = new Matrix<float>(channels, 1); InitMatrixFunc(*beta,  Zeros);
}

InstanceNorm2d::~InstanceNorm2d() {
    delete gamma; delete beta;
    delete dgamma; delete dbeta;
}

Matrix<float> InstanceNorm2d::forward(Matrix<float>& input) {
    int H = input.row, W = input.col, C = input.channel;
    int spatial = H * W;
    auto* out = new Matrix<float>(H, W, C);

    if (train_mode) gard.push_back(new Matrix<float>(input));

    // Per-sample per-channel: normalize each channel independently
    for (int c = 0; c < C; ++c) {
        float mean = 0;
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                mean += input.at(i, j, c);
        mean /= spatial;

        float var = 0;
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) {
                float d = input.at(i, j, c) - mean;
                var += d * d;
            }
        var /= spatial;

        float inv_std = 1.0f / std::sqrt(var + eps);
        float gm = gamma->at(c, 0), bt = beta->at(c, 0);
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                out->at(i, j, c) = gm * (input.at(i, j, c) - mean) * inv_std + bt;
    }

    output.push_back(out);
    return *out;
}

Matrix<float> InstanceNorm2d::backward(Matrix<float>& grad_output) {
    int H = grad_output.row, W = grad_output.col, C = grad_output.channel;
    int spatial = H * W;
    auto& x = *gard.back();
    dgamma = new Matrix<float>(C, 1);
    dbeta  = new Matrix<float>(C, 1);
    auto* gi = new Matrix<float>(H, W, C);

    for (int c = 0; c < C; ++c) {
        float mean = 0;
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                mean += x.at(i, j, c);
        mean /= spatial;

        float var = 0;
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) {
                float d = x.at(i, j, c) - mean;
                var += d * d;
            }
        var /= spatial;

        float inv_std = 1.0f / std::sqrt(var + eps);

        float dx_hat_sum = 0, dx_hat_x_sum = 0;
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                float dx_hat = grad_output.at(i, j, c) * gamma->at(c, 0);
                dx_hat_sum += dx_hat;
                dx_hat_x_sum += dx_hat * (x.at(i, j, c) - mean);
                dgamma->at(c, 0) += grad_output.at(i, j, c) * (x.at(i, j, c) - mean) * inv_std;
                dbeta->at(c, 0)  += grad_output.at(i, j, c);
            }
        }
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                float dx_hat = grad_output.at(i, j, c) * gamma->at(c, 0);
                gi->at(i, j, c) = inv_std / spatial * (spatial * dx_hat - dx_hat_sum
                    - (x.at(i, j, c) - mean) * inv_std * inv_std * dx_hat_x_sum);
            }
        }
    }
    return *gi;
}

} // namespace NPCore
