#ifndef COREPP_LAYERS_MODULE_H
#define COREPP_LAYERS_MODULE_H

#include <vector>
#include "Core/Matrix.hpp"

namespace CoreNNSpace {

// =================================[Module 抽象基类]================================
// 所有神经网络层的基类，采用虚基类多态设计

template<typename T>
class Module {
protected:
    std::vector<Matrix<T>*> gard;
    std::vector<Matrix<T>*> output;

    // 预计算的权重/偏置梯度（由 backward() 填充，优化器读取）
    Matrix<T>* weight_grad_ = nullptr;
    Matrix<T>* bias_grad_ = nullptr;

public:
    // 动量（与梯度的作用类似）
    Matrix<T> m, v;

    // Train/eval mode
    bool train_mode = true;
    void eval()  { train_mode = false; }
    void train() { train_mode = true;  }

    // GPU: move all parameters to GPU / back to CPU
    virtual void cuda() { for (auto* p : getParams()) p->to(Device::GPU); }
    virtual void cpu()  { for (auto* p : getParams()) p->to(Device::CPU); }

    Module() = default;
    virtual ~Module();

    // 前向传播
    virtual Matrix<T> forward(Matrix<T> &input) = 0;

    // 反向传播: 给定损失对输出的梯度, 返回损失对输入的梯度
    // 同时将权重/偏置梯度存储在 weight_grad_ / bias_grad_ 中
    virtual Matrix<T> backward(Matrix<T>& grad_output) = 0;

    // 获取参数
    virtual std::vector<Matrix<float> *> getParams() = 0;

    // 获取所有梯度 (与 getParams 顺序对应, RNN/LSTM 有多个权重矩阵)
    virtual std::vector<Matrix<float>*> getAllGrads() {
        return {weight_grad_, bias_grad_};
    }

    // 获取存储的梯度（激活层返回导数，参数层返回输入）
    virtual Matrix<T>* getGard() = 0;

    // 获取输出
    virtual Matrix<T>* getOutput() = 0;

    // 获取预计算的权重梯度（backward 后有效）
    Matrix<T>* getWeightGrad() { return weight_grad_; }

    // 获取预计算的偏置梯度（backward 后有效）
    Matrix<T>* getBiasGrad() { return bias_grad_; }

    // 梯度清除
    virtual void CleanGard() = 0;
};

} // namespace CoreNNSpace

#endif // COREPP_LAYERS_MODULE_H
