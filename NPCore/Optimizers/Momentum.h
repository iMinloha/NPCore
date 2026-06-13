// Momentum.h - declaration only
#pragma once
#include "Optimizers/Optimizer.h"
#include <unordered_map>
namespace NPCore {
void Momentum_method(std::vector<Module<float>*>, Matrix<float>&, float, OptimState&, float = 0.9f);
}
