#pragma once
#include <unordered_map>
#include "Optimizers/Optimizer.h"
namespace CoreNNSpace {
inline void Momentum_method(std::vector<Module<float>*> params,Matrix<float>& grad,float lr,OptimState& st,float m=0.9f){
    Matrix<float> grad_=grad;auto& mv=st.momentum_velocity;
    for(int i=params.size()-1;i>=0;--i){auto*l=params[i];grad_=l->backward(grad_);
        if(l->getParams().empty()){l->CleanGard();continue;}
        auto pl=l->getParams(),gl=l->getAllGrads();
        for(size_t p=0;p<pl.size()&&p<gl.size();++p){auto*P=pl[p],*G=gl[p];if(!G)continue;
            if(mv.find(P)==mv.end())mv[P]=Matrix<float>(P->row,P->col);
            mv[P]=mv[P]*m+(*G)*lr;*P=*P-mv[P];}
        l->CleanGard();}}
}
