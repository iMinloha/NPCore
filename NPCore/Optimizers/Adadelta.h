// Adadelta.h - declaration only
#pragma once
#include "Optimizers/Optimizer.h"
#include <cmath>
namespace NPCore {
void Adadelta_method(std::vector<Module<float>*>, Matrix<float>&, float = 0.9f, float = 1e-6f);
}
