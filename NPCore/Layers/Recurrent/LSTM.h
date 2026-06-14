#ifndef NPCORE_LAYERS_LSTM_H
#define NPCORE_LAYERS_LSTM_H

#include <vector>
#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace NPCore {

// =================================[LSTM 长短期记忆网络]================================

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
    ~LSTM() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;

    std::vector<Matrix<float>*> getParams() override;
    std::vector<Matrix<float>*> getAllGrads() override;

    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;

    void CleanGard() override;
};

} // namespace NPCore

#endif
