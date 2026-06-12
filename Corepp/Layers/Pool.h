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
} // namespace
#endif
