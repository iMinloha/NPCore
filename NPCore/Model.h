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
NPCORE_API Module<float>* ReLU();
NPCORE_API Module<float>* Sigmoid();
NPCORE_API Module<float>* SoftMax();
NPCORE_API Module<float>* Tanh();
NPCORE_API Module<float>* Flatten();

// =================================[FNN / CNN builders]================================
NPCORE_API Sequence FNN(std::initializer_list<int> sizes,
                        Module<float>* (*activation)() = Sigmoid);
NPCORE_API Sequence CNN(std::initializer_list<int> channels,
                        int kernel = 3,
                        Module<float>* (*activation)() = Sigmoid,
                        int output_classes = 4);

// =================================[Loss function shorthands]================================
enum LossType { MSE, CrossEntropy };

NPCORE_API Matrix<float> loss_grad(const Matrix<float>& pred, const Matrix<float>& target,
                                    LossType loss = MSE);
NPCORE_API float loss_val(const Matrix<float>& pred, const Matrix<float>& target,
                           LossType = MSE);

// =================================[Optimizer shorthands]================================
NPCORE_API Optim SGD(float lr = 0.01f);
NPCORE_API Optim Adam(float lr = 0.001f);
NPCORE_API Optim RMSProp(float lr = 0.01f);

// =================================[Trainer: Training Loop]================================
class NPCORE_API Trainer {
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
