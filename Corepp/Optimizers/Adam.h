#pragma once
#include <cmath>
#include "Optimizers/Optimizer.h"
namespace CoreNNSpace {
inline void Adam_method(std::vector<Module<float>*> params,Matrix<float>& grad,float lr,OptimState& st,float b1=0.9f,float b2=0.999f,float eps=1e-8f){
    st.adam_t++;int t=st.adam_t;float mc=1/(1-std::pow(b1,t)),vc=1/(1-std::pow(b2,t));Matrix<float> grad_=grad;
    for(int i=params.size()-1;i>=0;i--){auto*l=params[i];grad_=l->backward(grad_);
        if(l->getParams().empty()){l->CleanGard();continue;}
        auto pl=l->getParams(),gl=l->getAllGrads();
        if(l->m.row==0&&!pl.empty()){l->m=Matrix<float>(pl[0]->row,pl[0]->col);l->v=Matrix<float>(pl[0]->row,pl[0]->col);}
        for(size_t p=0;p<pl.size()&&p<gl.size();++p){auto*P=pl[p],*G=gl[p];if(!G)continue;
            if(p==0&&P->row==l->m.row&&P->col==l->m.col)P->fused_adam_update(*G,l->m,l->v,b1,b2,mc,vc,lr,eps);
            else *P=*P-(*G)*lr;}
        l->CleanGard();}}
}
