#pragma once
#include <cmath>
#include "Optimizers/Optimizer.h"
namespace CoreNNSpace {
inline void RAdam_method(std::vector<Module<float>*> params,Matrix<float>& grad,float lr,OptimState& st,float b1=0.9f,float b2=0.999f,float eps=1e-8f){
    st.adam_t++;int t=st.adam_t;float ri=2/(1-b2)-1,rt=ri-2*t*std::pow(b2,t)/(1-std::pow(b2,t));Matrix<float> grad_=grad;
    for(int i=params.size()-1;i>=0;i--){auto*l=params[i];grad_=l->backward(grad_);
        if(l->getParams().empty()){l->CleanGard();continue;}
        auto pl=l->getParams(),gl=l->getAllGrads();
        if(l->m.row==0&&!pl.empty()){l->m=Matrix<float>(pl[0]->row,pl[0]->col);l->v=Matrix<float>(pl[0]->row,pl[0]->col);}
        for(size_t p=0;p<pl.size()&&p<gl.size();++p){auto*P=pl[p],*G=gl[p];if(!G)continue;
            int n=P->row*P->col;
            for(int j=0;j<n;++j){float g=G->data_ptr()[j];
                l->m.data_ptr()[j]=b1*l->m.data_ptr()[j]+(1-b1)*g;
                l->v.data_ptr()[j]=b2*l->v.data_ptr()[j]+(1-b2)*g*g;
                float mh=l->m.data_ptr()[j]/(1-std::pow(b1,t));
                if(rt>4){float vh=std::sqrt(l->v.data_ptr()[j]/(1-std::pow(b2,t)))+eps;
                    float r=std::sqrt((rt-4)*(rt-2)*ri/((ri-4)*(ri-2)*rt));P->data_ptr()[j]-=lr*r*mh/vh;}
                else P->data_ptr()[j]-=lr*mh;}}
        l->CleanGard();}}
}
