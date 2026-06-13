#include "Model.h"

namespace NPCore {
namespace nn {

Module<float>* ReLU()    { return new Activation::ReLU(); }
Module<float>* Sigmoid() { return new Activation::Sigmoid(); }
Module<float>* SoftMax(){ return new Activation::SoftMax(); }
Module<float>* Tanh()    { return new Activation::Sigmoid(); }
Module<float>* Flatten() { return new NPCore::Flatten(); }

Sequence FNN(std::initializer_list<int> sizes, Module<float>* (*activation)()) {
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

Sequence CNN(std::initializer_list<int> channels, int kernel,
             Module<float>* (*activation)(), int output_classes) {
    std::vector<Module<float>*> layers;
    auto it = channels.begin();
    int prev = *it;
    for (++it; it != channels.end(); ++it) {
        layers.push_back(new Conv2d(prev, *it, kernel, 1, 0));
        layers.push_back(activation());
        prev = *it;
    }
    layers.push_back(new NPCore::Flatten());
    int H = 8;
    for (size_t i = 1; i < channels.size(); ++i)
        H = (H - kernel + 1);
    int flattened = H * H * prev;
    layers.push_back(new Linear(flattened, output_classes));
    return Sequence(layers);
}

Matrix<float> loss_grad(const Matrix<float>& pred, const Matrix<float>& target, LossType loss) {
    if (loss == CrossEntropy) return cross_entropy_loss_grad(pred, target);
    return mse_loss_grad(pred, target);
}

float loss_val(const Matrix<float>& pred, const Matrix<float>& target, LossType) {
    return mse_loss(pred, target);
}

Optim SGD(float lr)   { return Optim({}, NPCore::SGD, lr); }
Optim Adam(float lr) { return Optim({}, NPCore::Adam, lr); }
Optim RMSProp(float lr) { return Optim({}, NPCore::RMSProp, lr); }

} // namespace nn
} // namespace NPCore
