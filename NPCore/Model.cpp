#include "Model.h"

namespace NPCore {
namespace nn {

Module<float>* ReLU()    { return new Activation::ReLU(); }
Module<float>* Sigmoid() { return new Activation::Sigmoid(); }
Module<float>* SoftMax(){ return new Activation::SoftMax(); }
Module<float>* Tanh()    { return new Activation::Tanh(); }
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

float loss_val(const Matrix<float>& pred, const Matrix<float>& target, LossType loss) {
    if (loss == CrossEntropy) return cross_entropy_loss(pred, target);
    return mse_loss(pred, target);
}

Optim SGD(float lr)      { return Optim(SGD_step, {}, lr); }
Optim Momentum(float lr) { return Optim(Momentum_step, {}, lr); }
Optim Adam(float lr)     { return Optim(Adam_step, {}, lr); }
Optim RMSProp(float lr)  { return Optim(RMSProp_step, {}, lr); }
Optim Adagrad(float lr)  { return Optim(Adagrad_step, {}, lr); }
Optim Adadelta(float lr) { return Optim(Adadelta_step, {}, lr); }
Optim NAdam(float lr)    { return Optim(NAdam_step, {}, lr); }
Optim RAdam(float lr)    { return Optim(RAdam_step, {}, lr); }

// =================================[Trainer]================================
Trainer::Trainer(Sequence& model, LossType loss, Optim optim)
    : model_(&model), optim_(optim), loss_(loss) {
    // Inject leaf modules into optimizer so step() can update parameters
    optim_.set_params(model.modules());
}

void Trainer::bind(Optim optim) {
    optim_ = optim;
    // Re-bind leaf modules from current model
    if (model_) optim_.set_params(model_->modules());
}

void Trainer::fit(Matrix<float>& input, Matrix<float>& target, int epochs,
         std::function<void(int, float)> callback) {
    for (int e = 0; e < epochs; ++e) {
        Matrix<float> out = model_->forward(input);
        optim_.step(loss_grad(out, target, loss_));
        if (callback && (e % 50 == 0 || e == epochs - 1))
            callback(e, loss_val(out, target, loss_));
    }
}

void Trainer::fit(std::vector<std::pair<Matrix<float>*, Matrix<float>*>>& samples,
         int epochs,
         std::function<void(int, float)> callback) {
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

void Trainer::fit(DataLoader& loader, int epochs,
         std::function<void(int, float)> callback) {
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

} // namespace nn
} // namespace NPCore
