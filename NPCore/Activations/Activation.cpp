#include "Activations/Activation.h"
#include <cmath>
namespace NPCore { namespace Activation {

// =================================[Trivial method implementations]================================
// ReLU
std::vector<Matrix<float>*> ReLU::getParams() { return {}; }
Matrix<float>* ReLU::getGard() { return gard.empty() ? nullptr : gard.back(); }
Matrix<float>* ReLU::getOutput() { return output.empty() ? nullptr : output.back(); }
void ReLU::CleanGard() { for (auto p : gard) { delete p; } gard.clear(); for (auto p : output) { delete p; } output.clear(); }

// LeakyReLU
LeakyReLU::LeakyReLU(float a) : alpha(a) {}
std::vector<Matrix<float>*> LeakyReLU::getParams() { return {}; }
Matrix<float>* LeakyReLU::getGard() { return gard.empty() ? nullptr : gard.back(); }
Matrix<float>* LeakyReLU::getOutput() { return output.empty() ? nullptr : output.back(); }
void LeakyReLU::CleanGard() { for (auto p : gard) { delete p; } gard.clear(); for (auto p : output) { delete p; } output.clear(); }

// Sigmoid
std::vector<Matrix<float>*> Sigmoid::getParams() { return {}; }
Matrix<float>* Sigmoid::getGard() { return gard.empty() ? nullptr : gard.back(); }
Matrix<float>* Sigmoid::getOutput() { return output.empty() ? nullptr : output.back(); }
void Sigmoid::CleanGard() { for (auto p : gard) { delete p; } gard.clear(); for (auto p : output) { delete p; } output.clear(); }

// Tanh
std::vector<Matrix<float>*> Tanh::getParams() { return {}; }
Matrix<float>* Tanh::getGard() { return gard.empty() ? nullptr : gard.back(); }
Matrix<float>* Tanh::getOutput() { return output.empty() ? nullptr : output.back(); }
void Tanh::CleanGard() { for (auto p : gard) { delete p; } gard.clear(); for (auto p : output) { delete p; } output.clear(); }

// SoftMax
std::vector<Matrix<float>*> SoftMax::getParams() { return {}; }
Matrix<float>* SoftMax::getGard() { return gard.empty() ? nullptr : gard.back(); }
Matrix<float>* SoftMax::getOutput() { return output.empty() ? nullptr : output.back(); }
void SoftMax::CleanGard() { for (auto p : gard) { delete p; } gard.clear(); for (auto p : output) { delete p; } output.clear(); }

// ELU
ELU::ELU(float a) : alpha(a) {}
std::vector<Matrix<float>*> ELU::getParams() { return {}; }
Matrix<float>* ELU::getGard() { return gard.empty() ? nullptr : gard.back(); }
Matrix<float>* ELU::getOutput() { return output.empty() ? nullptr : output.back(); }
void ELU::CleanGard() { for (auto p : gard) { delete p; } gard.clear(); for (auto p : output) { delete p; } output.clear(); }

// SELU
std::vector<Matrix<float>*> SELU::getParams() { return {}; }
Matrix<float>* SELU::getGard() { return gard.empty() ? nullptr : gard.back(); }
Matrix<float>* SELU::getOutput() { return output.empty() ? nullptr : output.back(); }
void SELU::CleanGard() { for (auto p : gard) { delete p; } gard.clear(); for (auto p : output) { delete p; } output.clear(); }

// Softplus
std::vector<Matrix<float>*> Softplus::getParams() { return {}; }
Matrix<float>* Softplus::getGard() { return gard.empty() ? nullptr : gard.back(); }
Matrix<float>* Softplus::getOutput() { return output.empty() ? nullptr : output.back(); }
void Softplus::CleanGard() { for (auto p : gard) { delete p; } gard.clear(); for (auto p : output) { delete p; } output.clear(); }

// Mish
std::vector<Matrix<float>*> Mish::getParams() { return {}; }
Matrix<float>* Mish::getGard() { return gard.empty() ? nullptr : gard.back(); }
Matrix<float>* Mish::getOutput() { return output.empty() ? nullptr : output.back(); }
void Mish::CleanGard() { for (auto p : gard) { delete p; } gard.clear(); for (auto p : output) { delete p; } output.clear(); }

// GELU
std::vector<Matrix<float>*> GELU::getParams() { return {}; }
Matrix<float>* GELU::getGard() { return gard.empty() ? nullptr : gard.back(); }
Matrix<float>* GELU::getOutput() { return output.empty() ? nullptr : output.back(); }
void GELU::CleanGard() { for (auto p : gard) { delete p; } gard.clear(); for (auto p : output) { delete p; } output.clear(); }

// Swish
std::vector<Matrix<float>*> Swish::getParams() { return {}; }
Matrix<float>* Swish::getGard() { return gard.empty() ? nullptr : gard.back(); }
Matrix<float>* Swish::getOutput() { return output.empty() ? nullptr : output.back(); }
void Swish::CleanGard() { for (auto p : gard) { delete p; } gard.clear(); for (auto p : output) { delete p; } output.clear(); }

// =================================[Forward / Backward]================================

Matrix<float> ReLU::forward(Matrix<float>& in) {
    auto *r=new Matrix<float>(in.row,in.col,in.channel),*d=new Matrix<float>(in.row,in.col,in.channel);
    int n=in.row*in.col*in.channel;
    for(int i=0;i<n;++i){float v=in.data_ptr()[i];r->data_ptr()[i]=v>0?v:0;d->data_ptr()[i]=v>0?1:0;}
    gard.push_back(d);output.push_back(r);return*r;
} Matrix<float> ReLU::backward(Matrix<float>& g){return gard.empty()?g:(g&(*gard.back()));}

Matrix<float> LeakyReLU::forward(Matrix<float>& in) {
    auto *r=new Matrix<float>(in.row,in.col,in.channel),*d=new Matrix<float>(in.row,in.col,in.channel);
    int n=in.row*in.col*in.channel;
    for(int i=0;i<n;++i){float v=in.data_ptr()[i];r->data_ptr()[i]=v>0?v:alpha*v;d->data_ptr()[i]=v>0?1:alpha;}
    gard.push_back(d);output.push_back(r);return*r;
} Matrix<float> LeakyReLU::backward(Matrix<float>& g){return gard.empty()?g:(g&(*gard.back()));}

Matrix<float> Sigmoid::forward(Matrix<float>& in) {
    auto *r=new Matrix<float>(in.row,in.col,in.channel),*d=new Matrix<float>(in.row,in.col,in.channel);
    int n=in.row*in.col*in.channel;
    for(int i=0;i<n;++i){float v=in.data_ptr()[i];
        if(v>88){r->data_ptr()[i]=1;d->data_ptr()[i]=0;}
        else if(v<-88){r->data_ptr()[i]=0;d->data_ptr()[i]=0;}
        else{float s=1/(1+std::exp(-v));r->data_ptr()[i]=s;d->data_ptr()[i]=s*(1-s);}}
    gard.push_back(d);output.push_back(r);return*r;
} Matrix<float> Sigmoid::backward(Matrix<float>& g){return gard.empty()?g:(g&(*gard.back()));}

Matrix<float> Tanh::forward(Matrix<float>& in) {
    auto *r=new Matrix<float>(in.row,in.col,in.channel),*d=new Matrix<float>(in.row,in.col,in.channel);
    int n=in.row*in.col*in.channel;
    for(int i=0;i<n;++i){float t=std::tanh(in.data_ptr()[i]);r->data_ptr()[i]=t;d->data_ptr()[i]=1-t*t;}
    gard.push_back(d);output.push_back(r);return*r;
} Matrix<float> Tanh::backward(Matrix<float>& g){return gard.empty()?g:(g&(*gard.back()));}

Matrix<float> SoftMax::forward(Matrix<float>& in){
    auto*r=new Matrix<float>(in.row,in.col);
    for(int i=0;i<in.row;++i){float mx=-1e30f,sum=0;
        for(int j=0;j<in.col;++j){float v=in.at(i,j);if(v>mx)mx=v;}
        for(int j=0;j<in.col;++j)sum+=std::exp(in.at(i,j)-mx);
        for(int j=0;j<in.col;++j)r->at(i,j)=std::exp(in.at(i,j)-mx)/sum;}
    gard.push_back(new Matrix<float>(*r));output.push_back(r);return*r;
}
Matrix<float> SoftMax::backward(Matrix<float>& g){if(gard.empty())return g;
    auto& y=*gard.back();auto*dx=new Matrix<float>(g.row,g.col);
    for(int i=0;i<g.row;++i){float ws=0;for(int j=0;j<g.col;++j)ws+=g.at(i,j)*y.at(i,j);
        for(int j=0;j<g.col;++j)dx->at(i,j)=y.at(i,j)*(g.at(i,j)-ws);}return*dx;}

Matrix<float> ELU::forward(Matrix<float>& in){
    auto *r=new Matrix<float>(in.row,in.col,in.channel),*d=new Matrix<float>(in.row,in.col,in.channel);
    int n=in.row*in.col*in.channel;
    for(int i=0;i<n;++i){float v=in.data_ptr()[i];if(v>0){r->data_ptr()[i]=v;d->data_ptr()[i]=1;}
        else{float e=alpha*(std::exp(v)-1);r->data_ptr()[i]=e;d->data_ptr()[i]=e+alpha;}}
    gard.push_back(d);output.push_back(r);return*r;
} Matrix<float> ELU::backward(Matrix<float>& g){return gard.empty()?g:(g&(*gard.back()));}

Matrix<float> SELU::forward(Matrix<float>& in){
    const float la=1.0507009873554804934193349852946f,a=1.6732632423543772848170429916717f;
    auto *r=new Matrix<float>(in.row,in.col,in.channel),*d=new Matrix<float>(in.row,in.col,in.channel);
    int n=in.row*in.col*in.channel;
    for(int i=0;i<n;++i){float v=in.data_ptr()[i];
        if(v>0){r->data_ptr()[i]=la*v;d->data_ptr()[i]=la;}
        else{float e=a*(std::exp(v)-1);r->data_ptr()[i]=la*e;d->data_ptr()[i]=la*(e+a);}}
    gard.push_back(d);output.push_back(r);return*r;
} Matrix<float> SELU::backward(Matrix<float>& g){return gard.empty()?g:(g&(*gard.back()));}

Matrix<float> Softplus::forward(Matrix<float>& in){
    auto *r=new Matrix<float>(in.row,in.col,in.channel),*d=new Matrix<float>(in.row,in.col,in.channel);
    int n=in.row*in.col*in.channel;
    for(int i=0;i<n;++i){float v=in.data_ptr()[i];
        if(v>20){r->data_ptr()[i]=v;d->data_ptr()[i]=1;}
        else{float sp=std::log(1+std::exp(v));r->data_ptr()[i]=sp;d->data_ptr()[i]=1/(1+std::exp(-v));}}
    gard.push_back(d);output.push_back(r);return*r;
} Matrix<float> Softplus::backward(Matrix<float>& g){return gard.empty()?g:(g&(*gard.back()));}

Matrix<float> Mish::forward(Matrix<float>& in){
    auto *r=new Matrix<float>(in.row,in.col,in.channel),*d=new Matrix<float>(in.row,in.col,in.channel);
    int n=in.row*in.col*in.channel;
    for(int i=0;i<n;++i){float v=in.data_ptr()[i],sp=std::log(1+std::exp(v)),th=std::tanh(sp);
        r->data_ptr()[i]=v*th;float omt=1-th*th;d->data_ptr()[i]=th+v*omt/(1+std::exp(-v));}
    gard.push_back(d);output.push_back(r);return*r;
} Matrix<float> Mish::backward(Matrix<float>& g){return gard.empty()?g:(g&(*gard.back()));}

Matrix<float> GELU::forward(Matrix<float>& in){
    const float A=0.7978845608f,B=0.044715f;
    auto *r=new Matrix<float>(in.row,in.col,in.channel),*d=new Matrix<float>(in.row,in.col,in.channel);
    int n=in.row*in.col*in.channel;
    for(int i=0;i<n;++i){float x=in.data_ptr()[i],x3=x*x*x,t=std::tanh(A*(x+B*x3));
        r->data_ptr()[i]=0.5f*x*(1+t);d->data_ptr()[i]=0.5f*(1+t)+0.5f*x*A*(1+3*B*x*x)*(1-t*t);}
    gard.push_back(d);output.push_back(r);return*r;
} Matrix<float> GELU::backward(Matrix<float>& g){return gard.empty()?g:(g&(*gard.back()));}

Matrix<float> Swish::forward(Matrix<float>& in){
    auto *r=new Matrix<float>(in.row,in.col,in.channel),*d=new Matrix<float>(in.row,in.col,in.channel);
    int n=in.row*in.col*in.channel;
    for(int i=0;i<n;++i){float x=in.data_ptr()[i],s=1/(1+std::exp(-x));
        r->data_ptr()[i]=x*s;d->data_ptr()[i]=s*(1+x*(1-s));}
    gard.push_back(d);output.push_back(r);return*r;
} Matrix<float> Swish::backward(Matrix<float>& g){return gard.empty()?g:(g&(*gard.back()));}

}}
