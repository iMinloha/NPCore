#ifndef NPCORE_OPTIMIZERS_OPTIMIZER_H
#define NPCORE_OPTIMIZERS_OPTIMIZER_H
#include <unordered_map>
#include <vector>
#include "Layers/Module.h"

namespace NPCore {

typedef enum {SGD,Momentum,Adam,RMSProp,Adagrad,Adadelta,NAdam,RAdam,AdamW} Optimizer;

struct OptimState {
    int adam_t=0;
    std::unordered_map<Matrix<float>*,Matrix<float>> momentum_velocity;
};

class Optim {
public:
    Optimizer optimizerMethod;
private:
    float learn_rate=0.001;
    std::vector<Module<float>*> params;
    OptimState state;
public:
    Optim() = default;
    explicit Optim(Optimizer o);
    explicit Optim(std::vector<Module<float>*> p, Optimizer o = SGD, float lr = 0.001);
    void step(Matrix<float> loss);
};

} // namespace NPCore

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
