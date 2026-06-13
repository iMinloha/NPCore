// RMSProp.h - declaration only
#pragma once
#include "Optimizers/Optimizer.h"
#include <cmath>
namespace NPCore {
void RMSProp_method(std::vector<Module<float>*>, Matrix<float>&, float, float = 0.9f, float = 1e-5f);
}
