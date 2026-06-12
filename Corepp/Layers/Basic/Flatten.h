#ifndef COREPP_LAYERS_FLATTEN_H
#define COREPP_LAYERS_FLATTEN_H

#include <vector>
#include "Layers/Module.h"

namespace CoreNNSpace {

// =================================[Flatten 展平层]================================
// 将多维输入展平为 2D 列向量 (col=1), 用于 Conv2d -> Linear 之间

class Flatten : public Module<float> {
private:
    int original_row, original_col, original_channel;

public:
    Flatten() = default;
    ~Flatten() override = default;  // gard, output cleaned by ~Module()

    Matrix<float> forward(Matrix<float> &input) override;

    // 反向传播: 将展平后的梯度还原为原始形状
    Matrix<float> backward(Matrix<float>& grad_output) override;

    std::vector<Matrix<float>*> getParams() override { return {}; }

    Matrix<float>* getGard() override { return gard.empty() ? nullptr : gard.back(); }

    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }

    void CleanGard() override {
        for (auto ptr : gard) delete ptr;
        gard.clear();
        for (auto ptr : output) delete ptr;
        output.clear();
    }
};

} // namespace CoreNNSpace

#endif // COREPP_LAYERS_FLATTEN_H
