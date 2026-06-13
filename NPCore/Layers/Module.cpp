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

// 显式模板实例化
template class Module<float>;

} // namespace NPCore
