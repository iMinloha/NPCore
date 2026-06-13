#ifndef COREPP_MODEL_H
#define COREPP_MODEL_H

// =================================[CorePP High-Level API]================================
// Simplified model creation & training.  Build networks in one line.
//
//   #include "CorePP.h"
//   #include "Model.h"
//
//   auto net = nn::FNN({4, 8, 4}, nn::Sigmoid);
//   nn::Trainer(net, nn::MSE, nn::Adam(0.01))
//       .fit(input, target, 200);

#include "CorePP.h"
#include "DataLoader.h"
#include <functional>
#include <initializer_list>

namespace CoreNNSpace {
namespace nn {

// =================================[Activation shorthands]================================
inline Module<float>* ReLU()    { return new Activation::ReLU(); }
inline Module<float>* Sigmoid() { return new Activation::Sigmoid(); }
inline Module<float>* SoftMax(){ return new Activation::SoftMax(); }
inline Module<float>* Tanh()    { return new Activation::Sigmoid(); }  // alias
inline Module<float>* Flatten() { return new CoreNNSpace::Flatten(); }

// =================================[FNN: Feedforward Network]================================
// nn::FNN({4, 8, 4}, nn::Sigmoid)  ->  4->Sigmoid->8->Sigmoid->4
inline Sequence FNN(std::initializer_list<int> sizes,
                     Module<float>* (*activation)() = Sigmoid) {
    std::vector<Module<float>*> layers;
    auto it = sizes.begin();
    int prev = *it;
    for (++it; it != sizes.end(); ++it) {
        layers.push_back(new Linear(prev, *it));
        if (it + 1 != sizes.end()) layers.push_back(activation());
        prev = *it;
    }
    return Sequence(layers);
}

// =================================[CNN: Convolutional Network]================================
// nn::CNN({3, 4, 2}, 3, nn::Sigmoid)  ->  Conv(3->4,k3)->Sig->Conv(4->2,k3)->Sig->Flatten->Linear
inline Sequence CNN(std::initializer_list<int> channels,
                     int kernel = 3,
                     Module<float>* (*activation)() = Sigmoid,
                     int output_classes = 4) {
    std::vector<Module<float>*> layers;
    auto it = channels.begin();
    int prev = *it;
    for (++it; it != channels.end(); ++it) {
        layers.push_back(new Conv2d(prev, *it, kernel, 1, 0));
        layers.push_back(activation());
        prev = *it;
    }
    layers.push_back(new CoreNNSpace::Flatten());
    // Auto-compute Linear input size: (H-2*(kernel-1))^2 * last_channels
    int H = 8;  // default input size
    for (size_t i = 1; i < channels.size(); ++i)
        H = (H - kernel + 1);
    int flattened = H * H * prev;
    layers.push_back(new Linear(flattened, output_classes));
    return Sequence(layers);
}

// =================================[Loss function shorthands]================================
enum LossType { MSE, CrossEntropy };

inline Matrix<float> loss_grad(const Matrix<float>& pred, const Matrix<float>& target,
                                LossType loss = MSE) {
    if (loss == CrossEntropy) return cross_entropy_loss_grad(pred, target);
    return mse_loss_grad(pred, target);
}

inline float loss_val(const Matrix<float>& pred, const Matrix<float>& target,
                       LossType /*loss*/ = MSE) {
    return mse_loss(pred, target);
}

// =================================[Optimizer shorthands]================================
inline Optim SGD(float lr = 0.01f)   { return Optim({}, CoreNNSpace::SGD, lr); }
inline Optim Adam(float lr = 0.001f) { return Optim({}, CoreNNSpace::Adam, lr); }
inline Optim RMSProp(float lr = 0.01f) { return Optim({}, CoreNNSpace::RMSProp, lr); }

// =================================[Trainer: Training Loop]================================
template<typename ModelType>
class Trainer {
    ModelType* model_;
    Optim optim_;
    LossType loss_;

public:
    Trainer(ModelType& model, LossType loss, Optim optim)
        : model_(&model), optim_(optim), loss_(loss) {}

    // Re-bind optimizer params (call after model construction)
    void bind(Optim optim) { optim_ = optim; }

    // Single-input training
    void fit(Matrix<float>& input, Matrix<float>& target, int epochs,
             std::function<void(int, float)> callback = nullptr) {
        for (int e = 0; e < epochs; ++e) {
            Matrix<float> out = model_->forward(input);
            optim_.step(loss_grad(out, target, loss_));
            if (callback && (e % 50 == 0 || e == epochs - 1))
                callback(e, loss_val(out, target, loss_));
        }
    }

    // Multi-sample training
    void fit(std::vector<std::pair<Matrix<float>*, Matrix<float>*>>& samples,
             int epochs,
             std::function<void(int, float)> callback = nullptr) {
        for (int e = 0; e < epochs; ++e) {
            float total = 0;
            for (auto& [in, tgt] : samples) {
                Matrix<float> out = model_->forward(*in);
                optim_.step(loss_grad(out, *tgt, loss_));
                total += loss_val(out, *tgt, loss_);
            }
            if (callback && (e % 50 == 0 || e == epochs - 1))
                callback(e, total / samples.size());
        }
    }

    // DataLoader training - 兼容任意 DataLoader 子类
    void fit(DataLoader& loader, int epochs,
             std::function<void(int, float)> callback = nullptr) {
        for (int e = 0; e < epochs; ++e) {
            loader.reset();
            float total = 0; int count = 0;
            Matrix<float> x, y;
            while (loader.next_batch(x, y)) {
                Matrix<float> out = model_->forward(x);
                optim_.step(loss_grad(out, y, loss_));
                total += loss_val(out, y, loss_);
                count++;
            }
            if (callback && (e % 50 == 0 || e == epochs - 1))
                callback(e, count > 0 ? total / count : 0);
        }
    }
};

} // namespace nn
} // namespace CoreNNSpace

#endif // COREPP_MODEL_H
