#ifndef NPCORE_LAYERS_GRU_H
#define NPCORE_LAYERS_GRU_H
#include <vector>
#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace NPCore {

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
    GRU(int input_size, int hidden_size, InitMode mode = XavierUniform);
    ~GRU() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;

    std::vector<Matrix<float>*> getParams() override;
    std::vector<Matrix<float>*> getAllGrads() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;

    void CleanGard() override;
};

} // namespace
#endif
