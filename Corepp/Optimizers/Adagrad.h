#pragma once
#include <cmath>
#include "Optimizers/Optimizer.h"
namespace CoreNNSpace {
inline void Adagrad_method(std::vector<Module<float>*> params,Matrix<float>& grad,float lr,float eps=1e-5f){
    Matrix<float> grad_=grad;
    for(int i=params.size()-1;i>=0;i--){auto*l=params[i];grad_=l->backward(grad_);
        if(l->getParams().empty()){l->CleanGard();continue;}
        auto pl=l->getParams(),gl=l->getAllGrads();
        if(l->v.row==0&&!pl.empty())l->v=Matrix<float>(pl[0]->row,pl[0]->col);
        for(size_t p=0;p<pl.size()&&p<gl.size();++p){auto*P=pl[p],*G=gl[p];if(!G)continue;
            if(p==0&&P->row==l->v.row&&P->col==l->v.col)P->fused_adagrad_update(*G,l->v,lr,eps);
            else *P=*P-(*G)*lr;}
        l->CleanGard();}}
}
