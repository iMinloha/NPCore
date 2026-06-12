#pragma once
#include "Optimizers/Optimizer.h"
namespace CoreNNSpace {
inline void SGD_method(std::vector<Module<float>*> params,Matrix<float>& grad,float lr){
    Matrix<float> grad_=grad;
    for(int i=params.size()-1;i>=0;i--){auto*l=params[i];grad_=l->backward(grad_);
        if(l->getParams().empty()){l->CleanGard();continue;}
        auto pl=l->getParams(),gl=l->getAllGrads();
        for(size_t p=0;p<pl.size()&&p<gl.size();++p)if(gl[p])*pl[p]=*pl[p]-(*gl[p])*lr;
        l->CleanGard();}}
}
