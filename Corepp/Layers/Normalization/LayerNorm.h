#ifndef COREPP_LAYERS_LAYERNORM_H
#define COREPP_LAYERS_LAYERNORM_H
#include <cmath>
#include "Layers/Module.h"
#include "Layers/ParamInit.h"
namespace CoreNNSpace {

// =================================[LayerNorm]================================
// Normalize across feature dimension: y = gamma * (x - mu)/sqrt(sigma^2+eps) + beta
// Unlike BatchNorm, statistics are per-sample, no running averages needed.
class LayerNorm : public Module<float> {
    Matrix<float> *gamma, *beta, *dgamma = nullptr, *dbeta = nullptr;
    int features;
    float eps = 1e-5f;

public:
    LayerNorm(int features) : features(features) {
        gamma = new Matrix<float>(features, 1); InitMatrixFunc(*gamma, Ones);
        beta  = new Matrix<float>(features, 1); InitMatrixFunc(*beta,  Zeros);
    }
    ~LayerNorm() override { delete gamma; delete beta; delete dgamma; delete dbeta; }

    Matrix<float> forward(Matrix<float>& input) override {
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

    Matrix<float> backward(Matrix<float>& grad_output) override {
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

    std::vector<Matrix<float>*> getParams() override { return {gamma, beta}; }
    std::vector<Matrix<float>*> getAllGrads() override { return {dgamma, dbeta}; }
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override {
        for (auto p : gard) delete p;
        gard.clear();
        for (auto p : output) delete p;
        output.clear();
        delete dgamma; dgamma = nullptr;
        delete dbeta;  dbeta  = nullptr;
    }
};

} // namespace
#endif
