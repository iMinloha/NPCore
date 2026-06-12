#ifndef COREPP_OPTIMIZERS_LRSCHEDULER_H
#define COREPP_OPTIMIZERS_LRSCHEDULER_H
#include <cmath>
#include <functional>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace CoreNNSpace {

// =================================[Cosine Annealing LR Scheduler]================================
// lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T_max))
//
// Usage:
//   CosineLR sched(optim, 0.01f, 0.0001f, 500);  // lr_max=0.01, lr_min=1e-4, T=500
//   for (int e = 0; e < epochs; ++e) {
//       sched.step();    // updates optimizer's LR
//       train_one_epoch();
//   }

class CosineLR {
    float lr_max, lr_min, T_max;
    int t = 0;
    ::std::function<void(float)> set_lr;  // callback to set LR on optimizer

public:
    CosineLR(float lr_max, float lr_min, int T_max,
             ::std::function<void(float)> setter)
        : lr_max(lr_max), lr_min(lr_min), T_max(T_max), set_lr(setter) {}

    void step() {
        float lr = lr_min + 0.5f * (lr_max - lr_min) * (1.0f + ::std::cos(M_PI * t / T_max));
        set_lr(lr);
        t++;
    }
    float current_lr() const {
        return lr_min + 0.5f * (lr_max - lr_min) * (1.0f + ::std::cos(M_PI * t / T_max));
    }
    void reset() { t = 0; }
};

// =================================[Step Decay Scheduler]================================
class StepLR {
    float lr;
    int step_size;
    float gamma;
    int t = 0;
    ::std::function<void(float)> set_lr_;

public:
    StepLR(float lr, int step_size, float gamma, ::std::function<void(float)> setter)
        : lr(lr), step_size(step_size), gamma(gamma), set_lr_(setter) {}

    void step() {
        if (t > 0 && t % step_size == 0) lr *= gamma;
        set_lr_(lr);
        t++;
    }
};

} // namespace
#endif
