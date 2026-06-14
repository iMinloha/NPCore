#ifndef NPCORE_MODEL_H
#define NPCORE_MODEL_H

// =================================[NPCore High-Level API]================================
//   #include "NPCore.h"
//   #include "Model.h"
//   auto net = nn::FNN({4, 8, 4}, nn::Sigmoid);
//   nn::Trainer(net, nn::MSE, nn::Adam(0.01)).fit(input, target, 200);

#include "NPCore.h"
#include "DataLoader.h"
#include <functional>
#include <initializer_list>

namespace NPCore {
namespace nn {

// =================================[Activation shorthands]================================
Module<float>* ReLU();
Module<float>* Sigmoid();
Module<float>* SoftMax();
Module<float>* Tanh();
Module<float>* Flatten();

// =================================[FNN / CNN builders]================================
Sequence FNN(std::initializer_list<int> sizes,
             Module<float>* (*activation)() = Sigmoid);
Sequence CNN(std::initializer_list<int> channels,
             int kernel = 3,
             Module<float>* (*activation)() = Sigmoid,
             int output_classes = 4);

// =================================[Loss function shorthands]================================
enum LossType { MSE, CrossEntropy };

Matrix<float> loss_grad(const Matrix<float>& pred, const Matrix<float>& target,
                        LossType loss = MSE);
float loss_val(const Matrix<float>& pred, const Matrix<float>& target,
               LossType = MSE);

// =================================[Optimizer shorthands]================================
Optim SGD(float lr = 0.01f);
Optim Adam(float lr = 0.001f);
Optim RMSProp(float lr = 0.01f);

// =================================[Trainer: Training Loop]================================
class Trainer {
    Module<float>* model_;
    Optim optim_;
    LossType loss_;

public:
    Trainer(Module<float>& model, LossType loss, Optim optim);
    void bind(Optim optim);
    void fit(Matrix<float>& input, Matrix<float>& target, int epochs,
             std::function<void(int, float)> callback = nullptr);
    void fit(std::vector<std::pair<Matrix<float>*, Matrix<float>*>>& samples,
             int epochs,
             std::function<void(int, float)> callback = nullptr);
    void fit(DataLoader& loader, int epochs,
             std::function<void(int, float)> callback = nullptr);
};

} // namespace nn
} // namespace NPCore

#endif
