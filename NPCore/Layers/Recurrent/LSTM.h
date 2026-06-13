#ifndef NPCORE_LAYERS_LSTM_H
#define NPCORE_LAYERS_LSTM_H

#include <vector>
#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace NPCore {

// =================================[LSTM 长短期记忆网络]================================
// Placeholder - full implementation below

class LSTM : public Module<float> {
private:
    Matrix<float>* W;     // (4*hidden, input+hidden) merged weights
    Matrix<float>* b;     // (4*hidden, 1) merged bias
    int input_size, hidden_size;
    std::vector<Matrix<float>> h_cache, c_cache, x_cache, gate_cache;
    int seq_len = 0;
    float grad_clip = 10.0f;

    Matrix<float>* dW = nullptr;
    Matrix<float>* db = nullptr;

public:
    LSTM(int input_size, int hidden_size,
         InitMode mode = InitMode::XavierUniform);
    ~LSTM() override { delete W; delete b; delete dW; delete db; }

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
        h_cache.clear(); c_cache.clear(); x_cache.clear(); gate_cache.clear();
    }
};

} // namespace NPCore

#endif
