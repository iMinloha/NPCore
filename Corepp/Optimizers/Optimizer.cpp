#include <unordered_map>
#include <cmath>
#include <vector>
#include "Optimizers/Optimizer.h"

namespace CoreNNSpace {

// =================================[SGD 随机梯度下降]================================
void SGD_method(std::vector<Module<float>*> params,
                Matrix<float> &grad, float lr) {
    Matrix<float> grad_ = grad;

    for (int i = params.size() - 1; i >= 0; i--) {
        Module<float>* layer = params[i];

        // 反向传播: 计算梯度并存储 weight/bias grad
        grad_ = layer->backward(grad_);

        if (layer->getParams().empty()) {
            layer->CleanGard();
            continue;
        }

        auto param_list = layer->getParams();
        auto grad_list  = layer->getAllGrads();
        for (size_t p = 0; p < param_list.size() && p < grad_list.size(); ++p) {
            if (grad_list[p] != nullptr)
                *param_list[p] = *param_list[p] - (*grad_list[p]) * lr;
        }

        layer->CleanGard();
    }
}

// =================================[Momentum 动量法]================================
void Momentum_method(std::vector<Module<float>*> params,
                     Matrix<float>& grad, float lr, OptimState& state, float momentum=0.9) {
    Matrix<float> grad_ = grad;
    auto& momentum_velocity = state.momentum_velocity;

    for (int i = params.size() - 1; i >= 0; --i) {
        Module<float>* layer = params[i];

        // 反向传播
        grad_ = layer->backward(grad_);

        if (layer->getParams().empty()) {
            layer->CleanGard();
            continue;
        }

        auto param_list = layer->getParams();
        auto grad_list  = layer->getAllGrads();

        for (size_t p = 0; p < param_list.size() && p < grad_list.size(); ++p) {
            Matrix<float>* param_p = param_list[p];
            Matrix<float>* grad_p  = grad_list[p];
            if (grad_p == nullptr || param_p == nullptr) continue;

            if (momentum_velocity.find(param_p) == momentum_velocity.end())
                momentum_velocity[param_p] = Matrix<float>(param_p->row, param_p->col);

            momentum_velocity[param_p] = momentum_velocity[param_p] * momentum + (*grad_p) * lr;
            *param_p = *param_p - momentum_velocity[param_p];
        }

        layer->CleanGard();
    }
}

// =================================[Adam 自适应矩估计]================================
void Adam_method(std::vector<Module<float>*> params,
                 Matrix<float> &grad, float lr, OptimState& state, float beta1=0.9,
                 float beta2=0.999, float epsilon=1e-5) {
    state.adam_t += 1;
    int t = state.adam_t;
    float m_corr = 1.0f / (1.0f - std::pow(beta1, t));
    float v_corr = 1.0f / (1.0f - std::pow(beta2, t));

    Matrix<float> grad_ = grad;

    for (int i = params.size() - 1; i >= 0; i--) {
        Module<float>* layer = params[i];
        grad_ = layer->backward(grad_);

        if (layer->getParams().empty()) {
            layer->CleanGard();
            continue;
        }

        auto param_list = layer->getParams();
        auto grad_list  = layer->getAllGrads();

        // Initialize m/v to match first param shape
        if (layer->m.row == 0 && param_list.size() > 0) {
            layer->m = Matrix<float>(param_list[0]->row, param_list[0]->col);
            layer->v = Matrix<float>(param_list[0]->row, param_list[0]->col);
        }

        for (size_t p = 0; p < param_list.size() && p < grad_list.size(); ++p) {
            Matrix<float>* param_p = param_list[p];
            Matrix<float>* grad_p  = grad_list[p];
            if (grad_p == nullptr || param_p == nullptr) continue;

            // First param: fused Adam update. Others: simple SGD.
            if (p == 0 && param_p->row == layer->m.row && param_p->col == layer->m.col) {
                param_p->fused_adam_update(*grad_p, layer->m, layer->v,
                                            beta1, beta2, m_corr, v_corr, lr, epsilon);
            } else {
                *param_p = *param_p - (*grad_p) * lr;
            }
        }

        layer->CleanGard();
    }
}

// =================================[RMSProp 均方根传播]================================
void RMSProp_method(std::vector<Module<float>*> params,
                    Matrix<float> &grad, float lr, float beta=0.9,
                    float epsilon=1e-5) {
    Matrix<float> grad_ = grad;

    for (int i = params.size() - 1; i >= 0; i--) {
        Module<float>* layer = params[i];
        grad_ = layer->backward(grad_);

        if (layer->getParams().empty()) {
            layer->CleanGard();
            continue;
        }

        auto param_list = layer->getParams();
        auto grad_list  = layer->getAllGrads();

        if (layer->v.row == 0 && param_list.size() > 0) {
            layer->v = Matrix<float>(param_list[0]->row, param_list[0]->col);
        }

        for (size_t p = 0; p < param_list.size() && p < grad_list.size(); ++p) {
            Matrix<float>* param_p = param_list[p];
            Matrix<float>* grad_p  = grad_list[p];
            if (grad_p == nullptr || param_p == nullptr) continue;

            if (p == 0 && param_p->row == layer->v.row && param_p->col == layer->v.col) {
                param_p->fused_rmsprop_update(*grad_p, layer->v, beta, lr, epsilon);
            } else {
                *param_p = *param_p - (*grad_p) * lr;
            }
        }

        layer->CleanGard();
    }
}

// =================================[Adagrad 自适应梯度]================================
void Adagrad_method(std::vector<Module<float>*> params,
                    Matrix<float> &grad, float lr, float epsilon=1e-5) {
    Matrix<float> grad_ = grad;

    for (int i = params.size() - 1; i >= 0; i--) {
        Module<float>* layer = params[i];
        grad_ = layer->backward(grad_);

        if (layer->getParams().empty()) {
            layer->CleanGard();
            continue;
        }

        auto param_list = layer->getParams();
        auto grad_list  = layer->getAllGrads();

        if (layer->v.row == 0 && param_list.size() > 0) {
            layer->v = Matrix<float>(param_list[0]->row, param_list[0]->col);
        }

        for (size_t p = 0; p < param_list.size() && p < grad_list.size(); ++p) {
            Matrix<float>* param_p = param_list[p];
            Matrix<float>* grad_p  = grad_list[p];
            if (grad_p == nullptr || param_p == nullptr) continue;

            if (p == 0 && param_p->row == layer->v.row && param_p->col == layer->v.col) {
                param_p->fused_adagrad_update(*grad_p, layer->v, lr, epsilon);
            } else {
                *param_p = *param_p - (*grad_p) * lr;
            }
        }

        layer->CleanGard();
    }
}

// =================================[Optim::step 分发器]================================
void Optim::step(Matrix<float> loss) {
    // Gradient clipping: scale down if max exceeds threshold (prevents explosion)
    float loss_max = loss.max();
    if (loss_max > 50.0f) {
        float scale = 50.0f / loss_max;
        loss = loss * scale;
    }

    switch (optimizerMethod) {
        case SGD:
            SGD_method(params, loss, learn_rate);
            break;
        case Momentum:
            Momentum_method(params, loss, learn_rate, state);
            break;
        case Adam:
            Adam_method(params, loss, learn_rate, state);
            break;
        case RMSProp:
            RMSProp_method(params, loss, learn_rate);
            break;
        case Adagrad:
            Adagrad_method(params, loss, learn_rate);
            break;
        default:
            break;
    }
}

} // namespace CoreNNSpace
