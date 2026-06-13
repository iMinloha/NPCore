// Adagrad.h - declaration only
#pragma once
#include "Optimizers/Optimizer.h"
#include <cmath>
namespace NPCore {
void Adagrad_method(std::vector<Module<float>*>, Matrix<float>&, float, float = 1e-5f);
}
