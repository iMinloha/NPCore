#ifndef COREPP_LAYERS_LINEAR_H
#define COREPP_LAYERS_LINEAR_H

#include <vector>
#include "Layers/Module.h"
#include "Layers/ParamInit.h"

namespace CoreNNSpace {

// =================================[Linear 全连接层]================================
// y = W * x + b

class Linear : public Module<float> {
private:
    Matrix<float>* weight;
    Matrix<float>* bias;

public:
    Linear(int in_features, int out_features, InitMode mode = InitMode::Uniform,
           double mu = 0.0, double sigma = 1.0);

    ~Linear() override {
        delete weight;
        delete bias;
        // gard, output, weight_grad_, bias_grad_ cleaned by ~Module()
    }

    // 前向传播
    Matrix<float> forward(Matrix<float> &input) override;

    // 反向传播: 计算并存储 weight/bias 梯度, 返回输入梯度
    Matrix<float> backward(Matrix<float>& grad_output) override;

    // 获取参数
    std::vector<Matrix<float>*> getParams() override { return {weight, bias}; }

    // 获取梯度
    Matrix<float>* getGard() override { return gard.empty() ? nullptr : gard.back(); }

    // 获取输出
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }

    // 清除梯度
    void CleanGard() override {
        for (auto ptr : gard) delete ptr;
        gard.clear();
        for (auto ptr : output) delete ptr;
        output.clear();
        delete weight_grad_; weight_grad_ = nullptr;
        delete bias_grad_; bias_grad_ = nullptr;
    }
};

} // namespace CoreNNSpace

#endif // COREPP_LAYERS_LINEAR_H
