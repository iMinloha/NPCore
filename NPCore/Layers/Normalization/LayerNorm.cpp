#include "Layers/Normalization/LayerNorm.h"
#include <cmath>

namespace NPCore {

LayerNorm::LayerNorm(int features) : features(features) {
    gamma = new Matrix<float>(features, 1); InitMatrixFunc(*gamma, Ones);
    beta  = new Matrix<float>(features, 1); InitMatrixFunc(*beta,  Zeros);
}

LayerNorm::~LayerNorm() {
    delete gamma; delete beta;
    delete dgamma; delete dbeta;
}

Matrix<float> LayerNorm::forward(Matrix<float>& input) {
    int N = input.row, F = input.col;
    auto* out = new Matrix<float>(N, F);
    if (train_mode) gard.push_back(new Matrix<float>(input));

    for (int i = 0; i < N; ++i) {
        float mean = 0, var = 0;
        for (int f = 0; f < F; ++f) mean += input.at(i,f);
        mean /= F;
        for (int f = 0; f < F; ++f) { float d = input.at(i,f)-mean; var += d*d; }
        var /= F;
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int f = 0; f < F; ++f)
            out->at(i,f) = gamma->at(f,0) * (input.at(i,f) - mean) * inv_std + beta->at(f,0);
    }
    output.push_back(out); return *out;
}

Matrix<float> LayerNorm::backward(Matrix<float>& grad_output) {
    int N = grad_output.row, F = grad_output.col;
    auto& x = *gard.back();
    dgamma = new Matrix<float>(F,1); dbeta = new Matrix<float>(F,1);
    auto* gi = new Matrix<float>(N, F);

    for (int i = 0; i < N; ++i) {
        float mean = 0, var = 0;
        for (int f = 0; f < F; ++f) mean += x.at(i,f);
        mean /= F;
        for (int f = 0; f < F; ++f) { float d = x.at(i,f)-mean; var += d*d; }
        var /= F;
        float inv_std = 1.0f / std::sqrt(var + eps);

        float dx_hat_sum = 0, dx_hat_x_sum = 0;
        for (int f = 0; f < F; ++f) {
            float dx_hat = grad_output.at(i,f) * gamma->at(f,0);
            dx_hat_sum += dx_hat;
            dx_hat_x_sum += dx_hat * (x.at(i,f) - mean);
            dgamma->at(f,0) += grad_output.at(i,f) * (x.at(i,f)-mean) * inv_std;
            dbeta->at(f,0)  += grad_output.at(i,f);
        }
        for (int f = 0; f < F; ++f) {
            float dx_hat = grad_output.at(i,f) * gamma->at(f,0);
            gi->at(i,f) = inv_std / F * (F * dx_hat - dx_hat_sum
                - (x.at(i,f)-mean) * inv_std * inv_std * dx_hat_x_sum);
        }
    }
    return *gi;
}

} // namespace NPCore
