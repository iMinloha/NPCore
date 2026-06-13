#include "Layers/Normalization/BatchNorm.h"
#include <cmath>

namespace NPCore {

// =================================[BatchNorm1d]================================

BatchNorm1d::BatchNorm1d(int features) {
    gamma = new Matrix<float>(features, 1); InitMatrixFunc(*gamma, Ones);
    beta  = new Matrix<float>(features, 1); InitMatrixFunc(*beta,  Zeros);
    running_mean = new Matrix<float>(features, 1);
    running_var  = new Matrix<float>(features, 1);
}

BatchNorm1d::~BatchNorm1d() {
    delete gamma; delete beta;
    delete running_mean; delete running_var;
    delete dgamma; delete dbeta;
}

Matrix<float> BatchNorm1d::forward(Matrix<float>& input) {
    int N = input.row, F = input.col;
    auto* out = new Matrix<float>(N, F);

    // Compute mean/var
    Matrix<float> mean(F,1), var(F,1);
    for (int f = 0; f < F; ++f) {
        float s = 0; for (int i = 0; i < N; ++i) s += input.at(i,f);
        mean.at(f,0) = s / N;
        float sv = 0; for (int i = 0; i < N; ++i) { float d=input.at(i,f)-mean.at(f,0); sv+=d*d; }
        var.at(f,0) = sv / N;
    }
    if (train_mode && track_running) {
        for (int f=0;f<F;++f) { running_mean->at(f,0)=momentum*running_mean->at(f,0)+(1-momentum)*mean.at(f,0);
                                 running_var->at(f,0)=momentum*running_var->at(f,0)+(1-momentum)*var.at(f,0); }
    }
    auto& m = train_mode ? mean : *running_mean;
    auto& v = train_mode ? var  : *running_var;

    // Normalize
    if (train_mode) { gard.push_back(new Matrix<float>(input));
                      gard.push_back(new Matrix<float>(mean));
                      gard.push_back(new Matrix<float>(var)); }
    for (int i = 0; i < N; ++i)
        for (int f = 0; f < F; ++f)
            out->at(i,f) = gamma->at(f,0) * (input.at(i,f)-m.at(f,0)) / std::sqrt(v.at(f,0)+eps) + beta->at(f,0);
    output.push_back(out); return *out;
}

Matrix<float> BatchNorm1d::backward(Matrix<float>& grad_output) {
    int N = grad_output.row, F = grad_output.col;
    auto& x = *gard[gard.size()-3]; auto& mean = *gard[gard.size()-2]; auto& var = *gard[gard.size()-1];
    dgamma = new Matrix<float>(F,1); dbeta = new Matrix<float>(F,1);
    auto* gi = new Matrix<float>(N, F);

    for (int f = 0; f < F; ++f) {
        float inv_std = 1.0f / std::sqrt(var.at(f,0)+eps);
        float dx_hat_sum = 0, dx_hat_x_centered_sum = 0;
        for (int i = 0; i < N; ++i) {
            float dx_hat = grad_output.at(i,f) * gamma->at(f,0);
            dx_hat_sum += dx_hat;
            dx_hat_x_centered_sum += dx_hat * (x.at(i,f) - mean.at(f,0));
            dgamma->at(f,0) += grad_output.at(i,f) * (x.at(i,f)-mean.at(f,0)) * inv_std;
            dbeta->at(f,0)  += grad_output.at(i,f);
        }
        for (int i = 0; i < N; ++i) {
            gi->at(i,f) = inv_std / N * (N * grad_output.at(i,f) * gamma->at(f,0)
                - dx_hat_sum - (x.at(i,f)-mean.at(f,0)) * inv_std * inv_std * dx_hat_x_centered_sum);
        }
    }
    return *gi;
}

void BatchNorm1d::CleanGard() {
    for (auto p : gard) { delete p; }
    gard.clear();
    for (auto p : output) { delete p; }
    output.clear();
    delete dgamma; dgamma=nullptr;
    delete dbeta; dbeta=nullptr;
}

// =================================[BatchNorm2d]================================

BatchNorm2d::BatchNorm2d(int channels) : channels(channels) {
    gamma = new Matrix<float>(channels, 1);
    beta  = new Matrix<float>(channels, 1);
    running_mean = new Matrix<float>(channels, 1);
    running_var  = new Matrix<float>(channels, 1);
    InitMatrixFunc(*gamma, Ones);
    InitMatrixFunc(*beta,  Zeros);
}

BatchNorm2d::~BatchNorm2d() {
    delete gamma; delete beta;
    delete running_mean; delete running_var;
    delete dgamma; delete dbeta;
}

Matrix<float> BatchNorm2d::forward(Matrix<float>& input) {
    int H = input.row, W = input.col, C = input.channel;
    int spatial = H * W;
    auto* out = new Matrix<float>(H, W, C);

    // Compute per-channel mean and variance over spatial dims
    Matrix<float> mean(C, 1), var(C, 1);
    for (int c = 0; c < C; ++c) {
        float sum = 0.0f;
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                sum += input.at(i, j, c);
        mean.at(c, 0) = sum / spatial;

        float sum_sq = 0.0f;
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j) {
                float d = input.at(i, j, c) - mean.at(c, 0);
                sum_sq += d * d;
            }
        var.at(c, 0) = sum_sq / spatial;
    }

    // Update running statistics (train mode)
    if (train_mode && track_running) {
        for (int c = 0; c < C; ++c) {
            running_mean->at(c, 0) = momentum * running_mean->at(c, 0)
                + (1.0f - momentum) * mean.at(c, 0);
            running_var->at(c, 0) = momentum * running_var->at(c, 0)
                + (1.0f - momentum) * var.at(c, 0);
        }
    }

    auto& m = train_mode ? mean : *running_mean;
    auto& v = train_mode ? var  : *running_var;

    // Save for backward
    if (train_mode) {
        gard.push_back(new Matrix<float>(input));
        gard.push_back(new Matrix<float>(mean));
        gard.push_back(new Matrix<float>(var));
    }

    // Normalize
    for (int c = 0; c < C; ++c) {
        float inv_std = 1.0f / std::sqrt(v.at(c, 0) + eps);
        float gm = gamma->at(c, 0), bt = beta->at(c, 0);
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                out->at(i, j, c) = gm * (input.at(i, j, c) - m.at(c, 0)) * inv_std + bt;
    }

    output.push_back(out);
    return *out;
}

Matrix<float> BatchNorm2d::backward(Matrix<float>& grad_output) {
    int H = grad_output.row, W = grad_output.col, C = grad_output.channel;
    int spatial = H * W;

    auto& x = *gard[gard.size() - 3];
    auto& mean = *gard[gard.size() - 2];
    auto& var  = *gard[gard.size() - 1];

    dgamma = new Matrix<float>(C, 1);
    dbeta  = new Matrix<float>(C, 1);
    auto* gi = new Matrix<float>(H, W, C);

    for (int c = 0; c < C; ++c) {
        float inv_std = 1.0f / std::sqrt(var.at(c, 0) + eps);
        float gamma_c = gamma->at(c, 0);

        float dx_hat_sum = 0.0f, dx_hat_x_centered_sum = 0.0f;
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                float dx_hat = grad_output.at(i, j, c) * gamma_c;
                dx_hat_sum += dx_hat;
                dx_hat_x_centered_sum += dx_hat * (x.at(i, j, c) - mean.at(c, 0));
                dgamma->at(c, 0) += grad_output.at(i, j, c)
                    * (x.at(i, j, c) - mean.at(c, 0)) * inv_std;
                dbeta->at(c, 0) += grad_output.at(i, j, c);
            }
        }

        float scale = inv_std / spatial;
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                gi->at(i, j, c) = scale * (spatial * grad_output.at(i, j, c) * gamma_c
                    - dx_hat_sum
                    - (x.at(i, j, c) - mean.at(c, 0)) * inv_std * inv_std * dx_hat_x_centered_sum);
            }
        }
    }

    return *gi;
}

void BatchNorm2d::CleanGard() {
    for (auto p : gard) { delete p; }
    gard.clear();
    for (auto p : output) { delete p; }
    output.clear();
    delete dgamma; dgamma = nullptr;
    delete dbeta;  dbeta  = nullptr;
}

} // namespace NPCore
