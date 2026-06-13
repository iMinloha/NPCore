// RAdam.h - declaration only
#pragma once
#include "Optimizers/Optimizer.h"
#include <cmath>
namespace NPCore {
void RAdam_method(std::vector<Module<float>*>, Matrix<float>&, float, OptimState&, float = 0.9f, float = 0.999f, float = 1e-8f);
}
