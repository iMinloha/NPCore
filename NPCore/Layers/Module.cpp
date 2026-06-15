#include "Layers/Module.h"

namespace NPCore {

// 析构函数
template<typename T>
Module<T>::~Module() {
    for (auto &i : gard) delete i;
    gard.clear();
    for (auto &i : output) delete i;
    output.clear();
    delete weight_grad_;
    delete bias_grad_;
    weight_grad_ = nullptr;
    bias_grad_ = nullptr;
}

template<typename T>
void Module<T>::eval() { train_mode = false; }

template<typename T>
void Module<T>::train() { train_mode = true; }

template<typename T>
void Module<T>::cuda() { for (auto* p : getParams()) p->to(Device::GPU); }

template<typename T>
void Module<T>::cpu() { for (auto* p : getParams()) p->to(Device::CPU); }

template<typename T>
std::vector<Module<float>*> Module<T>::modules() { return {reinterpret_cast<Module<float>*>(this)}; }

template<typename T>
std::vector<Matrix<float>*> Module<T>::getAllGrads() {
    return {weight_grad_, bias_grad_};
}

template<typename T>
Matrix<T>* Module<T>::getWeightGrad() { return weight_grad_; }

template<typename T>
Matrix<T>* Module<T>::getBiasGrad() { return bias_grad_; }

// 显式模板实例化
template class Module<float>;

} // namespace NPCore
