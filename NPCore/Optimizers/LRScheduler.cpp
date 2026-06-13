#include "Optimizers/LRScheduler.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace NPCore {

CosineLR::CosineLR(float lr_max, float lr_min, int T_max,
                   std::function<void(float)> setter)
    : lr_max(lr_max), lr_min(lr_min), T_max(T_max), set_lr(setter) {}

void CosineLR::step() {
    float lr = lr_min + 0.5f * (lr_max - lr_min) * (1.0f + std::cos(M_PI * t / T_max));
    set_lr(lr);
    t++;
}

float CosineLR::current_lr() const {
    return lr_min + 0.5f * (lr_max - lr_min) * (1.0f + std::cos(M_PI * t / T_max));
}

void CosineLR::reset() { t = 0; }

StepLR::StepLR(float lr, int step_size, float gamma, std::function<void(float)> setter)
    : lr(lr), step_size(step_size), gamma(gamma), set_lr_(setter) {}

void StepLR::step() {
    if (t > 0 && t % step_size == 0) lr *= gamma;
    set_lr_(lr);
    t++;
}

} // namespace NPCore
