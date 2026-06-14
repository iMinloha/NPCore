#ifndef NPCORE_CORE_CONVUTILS_H
#define NPCORE_CORE_CONVUTILS_H

#include "Core/Matrix.hpp"

// =================================[im2col & col2im - Convolution as Matrix Multiply]================================
//
// These functions transform convolution into GEMM, reducing O(N^6) nested loops
// to O(matrix_multiply) using the already-optimized SIMD matmul in Matrix::operator*.
//
// im2col:  extracts sliding-window patches from a 3D input and arranges them as
//          columns of a 2D matrix.  Each column is one kernel-sized patch, flattened.
//          Output shape: (C_in * K_h * K_w, H_out * W_out)
//
// col2im:  inverse of im2col - scatters columns back to a 3D gradient tensor.
//          Used to compute dL/dX from dL/d(col).

namespace NPCore {

template<typename T>
Matrix<T> im2col(const Matrix<T>& input, int kernel_h, int kernel_w,
                 int stride, int padding);

template<typename T>
Matrix<T> col2im(const Matrix<T>& col, int H, int W, int C_in,
                 int kernel_h, int kernel_w, int stride, int padding);

} // namespace NPCore

// =================================[Winograd F(2x2, 3x3) Minimal Filtering]================================
//
// Reference: Lavin & Gray (2016) "Fast Algorithms for Convolutional Neural Networks"
//
// For 3x3 stride-1 convolution, Winograd F(2,3) reduces multiplications by 2.25x.

namespace NPCore {

void winograd_BT_d_B(const float d[16], float U[16]);
void winograd_G_g_GT(const float g[9], float V[16]);
void winograd_AT_M_A(const float M[16], float Y[4]);
void winograd_B_dU_BT(const float dU[16], float dD[16]);
void winograd_GT_dV_G(const float dV[16], float dG[9]);
void winograd_forward_3x3(const float* input, int H, int W, int C_in,
                          const float* V_packed, int C_out, float* output);
void winograd_transform_filter(const float* weight, int C_out, int C_in,
                               float* V_packed);

} // namespace NPCore

#endif // NPCORE_CORE_CONVUTILS_H
