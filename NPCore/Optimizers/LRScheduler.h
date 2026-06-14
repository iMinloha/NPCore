#ifndef NPCORE_OPTIMIZERS_LRSCHEDULER_H
#define NPCORE_OPTIMIZERS_LRSCHEDULER_H

#include <functional>
#include "Core/Export.h"

namespace NPCore {

class NPCORE_API CosineLR {
    float lr_max, lr_min, T_max;
    int t = 0;
    std::function<void(float)> set_lr;

public:
    CosineLR(float lr_max, float lr_min, int T_max,
             std::function<void(float)> setter);
    void step();
    float current_lr() const;
    void reset();
};

class NPCORE_API StepLR {
    float lr;
    int step_size;
    float gamma;
    int t = 0;
    std::function<void(float)> set_lr_;

public:
    StepLR(float lr, int step_size, float gamma, std::function<void(float)> setter);
    void step();
};

} // namespace NPCore
#endif
