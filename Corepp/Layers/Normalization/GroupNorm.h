#ifndef COREPP_LAYERS_GROUPNORM_H
#define COREPP_LAYERS_GROUPNORM_H

#include <cmath>
#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace CoreNNSpace {

// =================================[GroupNorm — Group Normalization]================================
// Reference: Wu & He (2018) "Group Normalization"
//
// Divides channels into groups and normalizes within each group over (H, W, C_per_group).
// Unlike BatchNorm, GroupNorm is independent of batch size — it works equally well
// with batch_size=1 or batch_size=32, making it ideal for:
//   - Object detection / segmentation (small batch sizes due to large inputs)
//   - Video models (temporal batch limits)
//   - Style transfer / GANs
//
// Input:  (H, W, C)
// Output: (H, W, C) — same shape
//
// y = γ_c * (x_c - μ_group(c)) / √(σ²_group(c) + ε) + β_c
//
// num_groups must divide channels (typically num_groups = 32 or channels // 2).

class GroupNorm : public Module<float> {
    int num_groups, channels, channels_per_group;
    Matrix<float> *gamma, *beta;
    Matrix<float> *dgamma = nullptr, *dbeta = nullptr;
    float eps = 1e-5f;

    // Helper: get group index for a channel
    int group_of(int ch) const { return ch / channels_per_group; }

public:
    GroupNorm(int num_groups, int channels)
        : num_groups(num_groups), channels(channels),
          channels_per_group(channels / num_groups)
    {
        COREPP_ASSERT(channels % num_groups == 0,
                    "channels (%d) must be divisible by num_groups (%d)", channels, num_groups);

        gamma = new Matrix<float>(channels, 1);
        beta  = new Matrix<float>(channels, 1);
        InitMatrixFunc(*gamma, Ones);
        InitMatrixFunc(*beta,  Zeros);
    }

    ~GroupNorm() override {
        delete gamma; delete beta;
        delete dgamma; delete dbeta;
    }

    Matrix<float> forward(Matrix<float>& input) override {
        int H = input.row, W = input.col, C = input.channel;
        int spatial = H * W;
        int cpg = channels_per_group;

        auto* out = new Matrix<float>(H, W, C);

        // Save input for backward
        if (train_mode) gard.push_back(new Matrix<float>(input));

        for (int g = 0; g < num_groups; ++g) {
            int ch_start = g * cpg;
            int ch_end   = ch_start + cpg;
            int N = spatial * cpg;  // total elements in group

            // Compute group mean
            float mean = 0.0f;
            for (int c = ch_start; c < ch_end; ++c)
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j)
                        mean += input.at(i, j, c);
            mean /= N;

            // Compute group variance
            float var = 0.0f;
            for (int c = ch_start; c < ch_end; ++c)
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j) {
                        float d = input.at(i, j, c) - mean;
                        var += d * d;
                    }
            var /= N;

            // Normalize + affine
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

    Matrix<float> backward(Matrix<float>& grad_output) override {
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

            // Compute group mean and var (forward pass values needed for backward)
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

            // dgamma and dbeta per channel
            for (int c = ch_start; c < ch_end; ++c) {
                for (int i = 0; i < H; ++i) {
                    for (int j = 0; j < W; ++j) {
                        dgamma->at(c, 0) += grad_output.at(i, j, c)
                            * (x.at(i, j, c) - mean) * inv_std;
                        dbeta->at(c, 0) += grad_output.at(i, j, c);
                    }
                }
            }

            // Gradient w.r.t. input (same derivation as LayerNorm but per-group)
            // For each position within the group:
            //   dx = (1/σ) * [ γ * dŷ - (γ * dŷ)_mean - x̂ * (γ * dŷ * x̂)_mean ] / N
            // where x̂ = (x - μ)/σ, dŷ = grad_output

            for (int c = ch_start; c < ch_end; ++c) {
                float gamma_c = gamma->at(c, 0);

                for (int i = 0; i < H; ++i) {
                    for (int j = 0; j < W; ++j) {
                        float x_hat = (x.at(i, j, c) - mean) * inv_std;
                        float dy_hat = grad_output.at(i, j, c) * gamma_c;

                        // Sum over the group: Σ dŷ and Σ dŷ * x̂
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

    std::vector<Matrix<float>*> getParams() override { return {gamma, beta}; }
    std::vector<Matrix<float>*> getAllGrads() override { return {dgamma, dbeta}; }
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }
    void CleanGard() override {
        for (auto p : gard) delete p;
        gard.clear();
        for (auto p : output) delete p;
        output.clear();
        delete dgamma; dgamma = nullptr;
        delete dbeta;  dbeta  = nullptr;
    }
};

} // namespace CoreNNSpace

#endif // COREPP_LAYERS_GROUPNORM_H
