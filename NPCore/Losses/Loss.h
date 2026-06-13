#ifndef NPCORE_LOSSES_H
#define NPCORE_LOSSES_H
#include <cmath>
#include <stdexcept>
#include "Core/Matrix.hpp"
namespace NPCore {

template<typename T> T mse_loss(const Matrix<T>& p,const Matrix<T>& t){
    if(p.row!=t.row||p.col!=t.col)throw std::runtime_error("MSE shape");
    T s=0;int n=p.row*p.col*p.channel;
    for(int i=0;i<n;++i){T d=p.data_ptr()[i]-t.data_ptr()[i];s+=d*d;}
    return s/(T)n;}
template<typename T> Matrix<T> mse_loss_grad(const Matrix<T>& p,const Matrix<T>& t){return p-t;}

template<typename T> T l1_loss(const Matrix<T>& p,const Matrix<T>& t){
    if(p.row!=t.row||p.col!=t.col)throw std::runtime_error("L1 shape");
    T s=0;int n=p.row*p.col*p.channel;
    for(int i=0;i<n;++i)s+=std::abs(p.data_ptr()[i]-t.data_ptr()[i]);
    return s/(T)n;}
template<typename T> Matrix<T> l1_loss_grad(const Matrix<T>& p,const Matrix<T>& t){
    Matrix<T> g(p.row,p.col,p.channel);int n=p.row*p.col*p.channel;
    for(int i=0;i<n;++i){T d=p.data_ptr()[i]-t.data_ptr()[i];g.data_ptr()[i]=d>0?1:(d<0?-1:0);}
    return g;}

template<typename T> T smooth_l1_loss(const Matrix<T>& p,const Matrix<T>& t,T b=1){
    if(p.row!=t.row||p.col!=t.col)throw std::runtime_error("Huber shape");
    T s=0;int n=p.row*p.col*p.channel;
    for(int i=0;i<n;++i){T d=std::abs(p.data_ptr()[i]-t.data_ptr()[i]);s+=d<b?0.5f*d*d/b:(d-0.5f*b);}
    return s/(T)n;}

template<typename T> Matrix<T> cross_entropy_loss_grad(const Matrix<T>& lg,const Matrix<T>& t){
    if(lg.row!=t.row||lg.col!=t.col)throw std::runtime_error("CE shape");
    Matrix<T> sm(lg.row,lg.col);
    for(int i=0;i<lg.row;++i){T mx=lg(i,0);for(int j=1;j<lg.col;++j)if(lg(i,j)>mx)mx=lg(i,j);
        T s=0;for(int j=0;j<lg.col;++j)s+=std::exp(lg(i,j)-mx);
        for(int j=0;j<lg.col;++j)sm.at(i,j)=std::exp(lg(i,j)-mx)/s;}
    return sm-t;}

template<typename T> T bce_loss(const Matrix<T>& p,const Matrix<T>& t){
    if(p.row!=t.row||p.col!=t.col)throw std::runtime_error("BCE shape");
    T s=0;int n=p.row*p.col*p.channel;T eps=1e-7f;
    for(int i=0;i<n;++i){T v=std::max(eps,std::min((T)1-eps,p.data_ptr()[i]));
        s-=t.data_ptr()[i]*std::log(v)+(1-t.data_ptr()[i])*std::log(1-v);}
    return s/(T)n;}

template<typename T> T kl_loss(const Matrix<T>& lp,const Matrix<T>& tp){
    if(lp.row!=tp.row||lp.col!=tp.col)throw std::runtime_error("KL shape");
    T s=0;int n=lp.row*lp.col*lp.channel;
    for(int i=0;i<n;++i)s+=tp.data_ptr()[i]*(std::log(tp.data_ptr()[i]+1e-10f)-lp.data_ptr()[i]);
    return s/(T)n;}

} // namespace
#endif
