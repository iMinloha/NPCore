#pragma once
#include <cmath>
#include "Optimizers/Optimizer.h"
namespace CoreNNSpace {
inline void Adadelta_method(std::vector<Module<float>*> params,Matrix<float>& grad,float rho=0.9f,float eps=1e-6f){
    Matrix<float> grad_=grad;
    for(int i=params.size()-1;i>=0;i--){auto*l=params[i];grad_=l->backward(grad_);
        if(l->getParams().empty()){l->CleanGard();continue;}
        auto pl=l->getParams(),gl=l->getAllGrads();
        if(l->v.row==0&&!pl.empty()){l->v=Matrix<float>(pl[0]->row,pl[0]->col);l->m=Matrix<float>(pl[0]->row,pl[0]->col);}
        for(size_t p=0;p<pl.size()&&p<gl.size();++p){auto*P=pl[p],*G=gl[p];if(!G)continue;
            int n=P->row*P->col;
            for(int j=0;j<n;++j){float g=G->data_ptr()[j];
                l->v.data_ptr()[j]=rho*l->v.data_ptr()[j]+(1-rho)*g*g;
                float dx=-std::sqrt(l->m.data_ptr()[j]+eps)/std::sqrt(l->v.data_ptr()[j]+eps)*g;
                l->m.data_ptr()[j]=rho*l->m.data_ptr()[j]+(1-rho)*dx*dx;P->data_ptr()[j]+=dx;}}
        l->CleanGard();}}
}
