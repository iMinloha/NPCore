#ifndef NPCORE_OPTIMIZERS_OPTIMIZER_H
#define NPCORE_OPTIMIZERS_OPTIMIZER_H
#include <unordered_map>
#include <vector>
#include "Layers/Module.h"

namespace NPCore {

// =================================[OptimState — shared state across optimizer steps]================================
struct OptimState {
    int adam_t = 0;
    std::unordered_map<Matrix<float>*, Matrix<float>> momentum_velocity;
};

// =================================[OptimStepFn — strategy function pointer]================================
// Every optimizer algorithm conforms to this signature.
using OptimStepFn = void(*)(std::vector<Module<float>*>, Matrix<float>&, float, OptimState&);

// =================================[Optim — lightweight wrapper]================================
class NPCORE_API Optim {
    OptimStepFn step_fn_ = nullptr;
    float learn_rate = 0.001f;
    std::vector<Module<float>*> params;
    OptimState state;

public:
    Optim() = default;
    Optim(OptimStepFn fn, std::vector<Module<float>*> p, float lr);
    void step(Matrix<float> loss);
};

// =================================[Factory functions]================================
NPCORE_API Optim SGD(float lr = 0.01f);
NPCORE_API Optim Adam(float lr = 0.001f);
NPCORE_API Optim RMSProp(float lr = 0.01f);

// =================================[Wrapper functions (match OptimStepFn)]================================
NPCORE_API void SGD_step(std::vector<Module<float>*>, Matrix<float>&, float, OptimState&);
NPCORE_API void Momentum_step(std::vector<Module<float>*>, Matrix<float>&, float, OptimState&);
NPCORE_API void Adam_step(std::vector<Module<float>*>, Matrix<float>&, float, OptimState&);
NPCORE_API void RMSProp_step(std::vector<Module<float>*>, Matrix<float>&, float, OptimState&);
NPCORE_API void Adagrad_step(std::vector<Module<float>*>, Matrix<float>&, float, OptimState&);
NPCORE_API void Adadelta_step(std::vector<Module<float>*>, Matrix<float>&, float, OptimState&);
NPCORE_API void NAdam_step(std::vector<Module<float>*>, Matrix<float>&, float, OptimState&);
NPCORE_API void RAdam_step(std::vector<Module<float>*>, Matrix<float>&, float, OptimState&);

} // namespace NPCore

// Backward-compatible sub-headers (internal use only)
#include "Optimizers/SGD.h"
#include "Optimizers/Momentum.h"
#include "Optimizers/Adam.h"
#include "Optimizers/RMSProp.h"
#include "Optimizers/Adagrad.h"
#include "Optimizers/Adadelta.h"
#include "Optimizers/NAdam.h"
#include "Optimizers/RAdam.h"
#include "Optimizers/AdamW.h"
#include "Optimizers/LRScheduler.h"

#endif
