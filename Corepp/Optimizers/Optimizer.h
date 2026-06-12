#ifndef COREPP_OPTIMIZERS_OPTIMIZER_H
#define COREPP_OPTIMIZERS_OPTIMIZER_H

#include <unordered_map>
#include <vector>
#include "Layers/Module.h"

namespace CoreNNSpace {

// =================================[Optimizer 优化器枚举]================================
typedef enum {
    SGD,        // 随机梯度下降
    Momentum,   // 动量法
    Adam,       // 自适应矩估计
    RMSProp,    // 均方根传播
    Adagrad,    // 自适应梯度
} Optimizer;

// =================================[OptimState 优化器实例状态]================================
struct OptimState {
    int adam_t = 0;  // Adam 步数计数器
    std::unordered_map<Matrix<float>*, Matrix<float>> momentum_velocity;  // Momentum 速度
};

// =================================[Optim 优化器类]================================
class Optim {
private:
    Optimizer optimizerMethod;
    float learn_rate = 0.001;
    std::vector<Module<float>*> params;
    OptimState state;

public:
    Optim() = default;

    explicit Optim(Optimizer optimizerMethod = SGD) : optimizerMethod(optimizerMethod) {}

    explicit Optim(std::vector<Module<float>*> params,
                   Optimizer optimizerMethod = SGD,
                   float learn_rate = 0.001)
        : optimizerMethod(optimizerMethod), learn_rate(learn_rate), params(std::move(params)) {}

    void step(Matrix<float> loss);
};

} // namespace CoreNNSpace

#endif // COREPP_OPTIMIZERS_OPTIMIZER_H
