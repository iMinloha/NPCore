#include "Optimizers/Optimizer.h"
namespace NPCore {

Optim::Optim(OptimStepFn fn, std::vector<Module<float>*> p, float lr)
    : step_fn_(fn), learn_rate(lr), params(std::move(p)) {}

void Optim::step(Matrix<float> loss) {
    float lm = loss.max();
    if (lm > 50.0f) loss = loss * (50.0f / lm);
    if (step_fn_) step_fn_(params, loss, learn_rate, state);
}

} // namespace NPCore
