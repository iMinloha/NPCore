#include "Optimizers/Optimizer.h"
namespace NPCore {

Optim::Optim(Optimizer o) : optimizerMethod(o) {}
Optim::Optim(std::vector<Module<float>*> p, Optimizer o, float lr)
    : optimizerMethod(o), learn_rate(lr), params(std::move(p)) {}

void Optim::step(Matrix<float> loss) {
    float lm=loss.max();
    if(lm>50.0f)loss=loss*(50.0f/lm);
    switch(optimizerMethod){
        case SGD:      SGD_method(params,loss,learn_rate);break;
        case Momentum: Momentum_method(params,loss,learn_rate,state);break;
        case Adam:     Adam_method(params,loss,learn_rate,state);break;
        case RMSProp:  RMSProp_method(params,loss,learn_rate);break;
        case Adagrad:  Adagrad_method(params,loss,learn_rate);break;
        case Adadelta: Adadelta_method(params,loss);break;
        case NAdam:    NAdam_method(params,loss,learn_rate,state);break;
        case RAdam:    RAdam_method(params,loss,learn_rate,state);break;
        default: break;
    }
}

} // namespace
