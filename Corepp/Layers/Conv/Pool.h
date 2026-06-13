#ifndef COREPP_LAYERS_POOL_H
#define COREPP_LAYERS_POOL_H
#include "Layers/Module.h"
namespace CoreNNSpace {

// =================================[MaxPool2d]================================
class MaxPool2d : public Module<float> {
    int pool_size, stride;
    int in_h, in_w, in_c;
    Matrix<float>* mask = nullptr; // argmax positions for backward

public:
    MaxPool2d(int pool_size = 2, int stride = 2) : pool_size(pool_size), stride(stride) {}
    ~MaxPool2d() override { delete mask; }

    Matrix<float> forward(Matrix<float>& input) override {
        in_h = input.row; in_w = input.col; in_c = input.channel;
        int out_h = (in_h - pool_size) / stride + 1;
        int out_w = (in_w - pool_size) / stride + 1;
        auto* out = new Matrix<float>(out_h, out_w, in_c);
        if (train_mode) mask = new Matrix<float>(out_h, out_w, in_c);

        for (int c = 0; c < in_c; ++c)
            for (int i = 0; i < out_h; ++i)
                for (int j = 0; j < out_w; ++j) {
                    float mx = -1e30f; int mi = 0, mj = 0;
                    for (int pi = 0; pi < pool_size; ++pi)
                        for (int pj = 0; pj < pool_size; ++pj) {
                            int si = i*stride+pi, sj = j*stride+pj;
                            float v = input.at(si,sj,c);
                            if (v > mx) { mx = v; mi = pi; mj = pj; }
                        }
                    out->at(i,j,c) = mx;
                    if (train_mode) mask->at(i,j,c) = (float)(mi * pool_size + mj);
                }
        output.push_back(out); return *out;
    }

    Matrix<float> backward(Matrix<float>& grad_output) override {
        int out_h = grad_output.row, out_w = grad_output.col;
        auto* gi = new Matrix<float>(in_h, in_w, in_c);
        for (int c = 0; c < in_c; ++c)
            for (int i = 0; i < out_h; ++i)
                for (int j = 0; j < out_w; ++j) {
                    int idx = (int)mask->at(i,j,c);
                    int pi = idx / pool_size, pj = idx % pool_size;
                    gi->at(i*stride+pi, j*stride+pj, c) = grad_output.at(i,j,c);
                }
        return *gi;
    }

    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override { for(auto p:gard)delete p; gard.clear(); for(auto p:output)delete p; output.clear(); delete mask; mask=nullptr; }
};
// =================================[AvgPool2d - Average Pooling]================================
// Averages over pooling windows instead of taking the max.
// Essential for many CNN variants (e.g., ResNet's final pooling before classifier).
//
// Input:  (H, W, C)
// Output: (H_out, W_out, C),  H_out = (H - pool_size)/stride + 1

class AvgPool2d : public Module<float> {
    int pool_size, stride;
    int in_h, in_w, in_c;

public:
    AvgPool2d(int pool_size = 2, int stride = 2) : pool_size(pool_size), stride(stride) {}
    ~AvgPool2d() override = default;

    Matrix<float> forward(Matrix<float>& input) override {
        in_h = input.row; in_w = input.col; in_c = input.channel;
        int out_h = (in_h - pool_size) / stride + 1;
        int out_w = (in_w - pool_size) / stride + 1;
        auto* out = new Matrix<float>(out_h, out_w, in_c);

        float scale = 1.0f / (pool_size * pool_size);

        for (int c = 0; c < in_c; ++c)
            for (int i = 0; i < out_h; ++i)
                for (int j = 0; j < out_w; ++j) {
                    float sum = 0.0f;
                    for (int pi = 0; pi < pool_size; ++pi)
                        for (int pj = 0; pj < pool_size; ++pj) {
                            int si = i * stride + pi, sj = j * stride + pj;
                            sum += input.at(si, sj, c);
                        }
                    out->at(i, j, c) = sum * scale;
                }
        output.push_back(out);
        return *out;
    }

    Matrix<float> backward(Matrix<float>& grad_output) override {
        int out_h = grad_output.row, out_w = grad_output.col;
        auto* gi = new Matrix<float>(in_h, in_w, in_c);
        float scale = 1.0f / (pool_size * pool_size);

        for (int c = 0; c < in_c; ++c)
            for (int i = 0; i < out_h; ++i)
                for (int j = 0; j < out_w; ++j) {
                    float g = grad_output.at(i, j, c) * scale;
                    for (int pi = 0; pi < pool_size; ++pi)
                        for (int pj = 0; pj < pool_size; ++pj)
                            gi->at(i * stride + pi, j * stride + pj, c) += g;
                }
        return *gi;
    }

    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }
    void CleanGard() override {
        for (auto p : gard) delete p;
        gard.clear();
        for (auto p : output) delete p;
        output.clear();
    }
};

// =================================[AdaptiveAvgPool2d - Adaptive Average Pooling]================================
// Pools to a fixed output size regardless of input dimensions.
// Essential for CNNs that must handle variable-size inputs (e.g., classifier head).
//
// Input:  (H, W, C)
// Output: (output_h, output_w, C) - fixed size
//
// Uses the standard adaptive pooling algorithm: for each output position,
// average over the corresponding region of the input.

class AdaptiveAvgPool2d : public Module<float> {
    int output_h, output_w;
    int in_h, in_w, in_c;

public:
    AdaptiveAvgPool2d(int output_h, int output_w)
        : output_h(output_h), output_w(output_w) {}

    // Convenience: single int -> square output (e.g., AdaptiveAvgPool2d(1) for global pooling)
    AdaptiveAvgPool2d(int output_size = 1)
        : output_h(output_size), output_w(output_size) {}

    ~AdaptiveAvgPool2d() override = default;

    Matrix<float> forward(Matrix<float>& input) override {
        in_h = input.row; in_w = input.col; in_c = input.channel;
        auto* out = new Matrix<float>(output_h, output_w, in_c);

        for (int c = 0; c < in_c; ++c) {
            for (int oh = 0; oh < output_h; ++oh) {
                // Input region for this output row
                int h_start = (int)std::floor((float)oh * in_h / output_h);
                int h_end   = (int)std::ceil((float)(oh + 1) * in_h / output_h);

                for (int ow = 0; ow < output_w; ++ow) {
                    int w_start = (int)std::floor((float)ow * in_w / output_w);
                    int w_end   = (int)std::ceil((float)(ow + 1) * in_w / output_w);

                    float sum = 0.0f;
                    int count = 0;
                    for (int ih = h_start; ih < h_end; ++ih) {
                        for (int iw = w_start; iw < w_end; ++iw) {
                            sum += input.at(ih, iw, c);
                            count++;
                        }
                    }
                    out->at(oh, ow, c) = sum / count;
                }
            }
        }
        output.push_back(out);
        return *out;
    }

    Matrix<float> backward(Matrix<float>& grad_output) override {
        auto* gi = new Matrix<float>(in_h, in_w, in_c);

        for (int c = 0; c < in_c; ++c) {
            for (int oh = 0; oh < output_h; ++oh) {
                int h_start = (int)std::floor((float)oh * in_h / output_h);
                int h_end   = (int)std::ceil((float)(oh + 1) * in_h / output_h);

                for (int ow = 0; ow < output_w; ++ow) {
                    int w_start = (int)std::floor((float)ow * in_w / output_w);
                    int w_end   = (int)std::ceil((float)(ow + 1) * in_w / output_w);

                    int count = (h_end - h_start) * (w_end - w_start);
                    float g = grad_output.at(oh, ow, c) / count;

                    for (int ih = h_start; ih < h_end; ++ih)
                        for (int iw = w_start; iw < w_end; ++iw)
                            gi->at(ih, iw, c) += g;
                }
            }
        }
        return *gi;
    }

    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }
    void CleanGard() override {
        for (auto p : gard) delete p;
        gard.clear();
        for (auto p : output) delete p;
        output.clear();
    }
};

} // namespace
#endif
