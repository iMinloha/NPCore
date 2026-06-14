#ifndef NPCORE_LOSSES_H
#define NPCORE_LOSSES_H

#include "Core/Matrix.hpp"

namespace NPCore {

template<typename T> T mse_loss(const Matrix<T>& p, const Matrix<T>& t);
template<typename T> Matrix<T> mse_loss_grad(const Matrix<T>& p, const Matrix<T>& t);

template<typename T> T l1_loss(const Matrix<T>& p, const Matrix<T>& t);
template<typename T> Matrix<T> l1_loss_grad(const Matrix<T>& p, const Matrix<T>& t);

template<typename T> T smooth_l1_loss(const Matrix<T>& p, const Matrix<T>& t, T b = 1);

template<typename T> Matrix<T> cross_entropy_loss_grad(const Matrix<T>& lg, const Matrix<T>& t);

template<typename T> T bce_loss(const Matrix<T>& p, const Matrix<T>& t);

template<typename T> T kl_loss(const Matrix<T>& lp, const Matrix<T>& tp);

} // namespace NPCore

#endif
