#ifndef COREPP_LAYERS_BATCHNORM_H
#define COREPP_LAYERS_BATCHNORM_H
#include <cmath>
#include "Layers/Module.h"
#include "Layers/ParamInit.h"
namespace CoreNNSpace {

// =================================[BatchNorm1d]================================
class BatchNorm1d : public Module<float> {
    Matrix<float> *gamma, *beta;
    Matrix<float> *running_mean, *running_var;
    Matrix<float> *dgamma, *dbeta;
    float momentum = 0.9f, eps = 1e-5f;
    bool track_running = true;

public:
    BatchNorm1d(int features) {
        gamma = new Matrix<float>(features, 1); InitMatrixFunc(*gamma, Ones);
        beta  = new Matrix<float>(features, 1); InitMatrixFunc(*beta,  Zeros);
        running_mean = new Matrix<float>(features, 1);
        running_var  = new Matrix<float>(features, 1);
    }
    ~BatchNorm1d() override { delete gamma; delete beta; delete running_mean; delete running_var; delete dgamma; delete dbeta; }

    Matrix<float> forward(Matrix<float>& input) override {
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

    Matrix<float> backward(Matrix<float>& grad_output) override {
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

    std::vector<Matrix<float>*> getParams() override { return {gamma, beta}; }
    std::vector<Matrix<float>*> getAllGrads() override { return {dgamma, dbeta}; }
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override { for(auto p:gard)delete p; gard.clear(); for(auto p:output)delete p; output.clear();
                                 delete dgamma; dgamma=nullptr; delete dbeta; dbeta=nullptr; }
};

} // namespace
#endif
