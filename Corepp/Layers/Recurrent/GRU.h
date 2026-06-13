#ifndef COREPP_LAYERS_GRU_H
#define COREPP_LAYERS_GRU_H
#include <vector>
#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace CoreNNSpace {

// =================================[GRU - Gated Recurrent Unit]================================
// r_t = sigma(W_ir·x + W_hr·h + b_r)    重置门
// z_t = sigma(W_iz·x + W_hz·h + b_z)    更新门
// n_t = tanh(W_in·x + r(*)(W_hn·h) + b_n)  新状态候选
// h_t = (1-z)(*)n + z(*)h_{t-1}          最终状态
//
// 合并权重: W(3H, I+H), b(3H, 1)

class GRU : public Module<float> {
    Matrix<float>* W, *b;
    int input_size, hidden_size, seq_len = 0;
    std::vector<Matrix<float>> h_cache, x_cache, gate_cache;
    Matrix<float>* dW = nullptr, *db = nullptr;
    float grad_clip = 10.0f;

public:
    GRU(int input_size, int hidden_size, InitMode mode = XavierUniform)
        : input_size(input_size), hidden_size(hidden_size) {
        int td = input_size + hidden_size;
        W = new Matrix<float>(3 * hidden_size, td);
        b = new Matrix<float>(3 * hidden_size, 1);
        InitMatrixFunc(*W, mode); InitMatrixFunc(*b, Zeros);
        for (int i = 0; i < 3 * hidden_size; ++i)
            for (int j = input_size; j < td; ++j) W->at(i, j) *= 0.1f;
        for (int i = 0; i < hidden_size; ++i) b->at(2*hidden_size + i, 0) = 2.0f; // update gate bias
    }
    ~GRU() override { delete W; delete b; delete dW; delete db; }

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;

    std::vector<Matrix<float>*> getParams() override { return {W, b}; }
    std::vector<Matrix<float>*> getAllGrads() override { return {dW, db}; }
    Matrix<float>* getGard() override { return gard.empty() ? nullptr : gard.back(); }
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }

    void CleanGard() override {
        for (auto p : gard) delete p;
        gard.clear();
        for (auto p : output) delete p;
        output.clear();
        delete dW; dW = nullptr;
        delete db; db = nullptr;
        h_cache.clear(); x_cache.clear(); gate_cache.clear();
    }
};

} // namespace
#endif
