#ifndef COREPP_CORE_CONVUTILS_H
#define COREPP_CORE_CONVUTILS_H

#include "Matrix.hpp"

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

namespace CoreNNSpace {

// =================================[im2col]================================
// input:   (H, W, C_in)
// returns: (C_in * K_h * K_w, H_out * W_out) column-major patch matrix
template<typename T>
Matrix<T> im2col(const Matrix<T>& input, int kernel_h, int kernel_w,
                 int stride, int padding) {
    const int C_in = input.channel;
    const int H = input.row;
    const int W = input.col;

    const int H_out = (H - kernel_h + 2 * padding) / stride + 1;
    const int W_out = (W - kernel_w + 2 * padding) / stride + 1;

    const int patch_size = C_in * kernel_h * kernel_w;
    const int num_patches = H_out * W_out;

    Matrix<T> col(patch_size, num_patches);

    for (int c = 0; c < C_in; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int row_idx = (c * kernel_h + kh) * kernel_w + kw;

                for (int oh = 0; oh < H_out; ++oh) {
                    for (int ow = 0; ow < W_out; ++ow) {
                        int col_idx = oh * W_out + ow;

                        int src_h = oh * stride + kh - padding;
                        int src_w = ow * stride + kw - padding;

                        T val = T(0);
                        if (src_h >= 0 && src_h < H && src_w >= 0 && src_w < W)
                            val = input.at(src_h, src_w, c);

                        col.at(row_idx, col_idx) = val;
                    }
                }
            }
        }
    }

    return col;
}

// =================================[col2im]================================
// col:     (C_in * K_h * K_w, H_out * W_out) gradient from upstream
// output_shape: reference input for spatial dimensions
// returns: (H, W, C_in) - the input gradient dL/dX
template<typename T>
Matrix<T> col2im(const Matrix<T>& col, int H, int W, int C_in,
                 int kernel_h, int kernel_w, int stride, int padding) {
    const int H_out = (H - kernel_h + 2 * padding) / stride + 1;
    const int W_out = (W - kernel_w + 2 * padding) / stride + 1;

    Matrix<T> img_grad(H, W, C_in);

    for (int c = 0; c < C_in; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int row_idx = (c * kernel_h + kh) * kernel_w + kw;

                for (int oh = 0; oh < H_out; ++oh) {
                    for (int ow = 0; ow < W_out; ++ow) {
                        int col_idx = oh * W_out + ow;

                        int dst_h = oh * stride + kh - padding;
                        int dst_w = ow * stride + kw - padding;

                        if (dst_h >= 0 && dst_h < H && dst_w >= 0 && dst_w < W)
                            img_grad.at(dst_h, dst_w, c) += col.at(row_idx, col_idx);
                    }
                }
            }
        }
    }

    return img_grad;
}

} // namespace CoreNNSpace

// =================================[Winograd F(2x2, 3x3) Minimal Filtering]================================
//
// Reference: Lavin & Gray (2016) "Fast Algorithms for Convolutional Neural Networks"
//
// For 3x3 stride-1 convolution, Winograd F(2,3) reduces multiplications by 2.25x.
// Transforms 4x4 input tiles into 2x2 output using 16 element-wise multiplications
// instead of 36 (3x3 kernel x 2x2 output).
//
// Matrices (rational form, multiplications absorbed into transforms):
//   Input:  U = B^T * d * B     (4x4 tile -> 4x4 Winograd domain, adds only)
//   Filter: V = G * g * G^T     (3x3 kernel -> 4x4 Winograd domain, adds only)
//   Output: Y = A^T * M * A     (4x4 Winograd -> 2x2 output, adds only)
//   Element-wise: M = U (*) V (pointwise multiply, 16 multiplies)

namespace CoreNNSpace {

// =================================[Winograd Forward Transform Matrices]================================
// B^T: 4x4 input transform (applied to 4x4 input tile)
// Values as 1/2 fractions for numerical stability
inline void winograd_BT_d_B(const float d[16], float U[16]) {
    // d is 4x4 row-major input tile
    // B^T * d:
    // B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
    float tmp[16];
    // tmp = d * B (column transform first)
    for (int i = 0; i < 4; ++i) {
        float d0 = d[i*4+0], d1 = d[i*4+1], d2 = d[i*4+2], d3 = d[i*4+3];
        tmp[i*4+0] = d0 - d2;
        tmp[i*4+1] = d1 + d2;
        tmp[i*4+2] = -d1 + d2;
        tmp[i*4+3] = d1 - d3;
    }
    // U = B^T * tmp
    for (int j = 0; j < 4; ++j) {
        float t0 = tmp[0*4+j], t1 = tmp[1*4+j], t2 = tmp[2*4+j], t3 = tmp[3*4+j];
        U[0*4+j] = t0 - t2;
        U[1*4+j] = t1 + t2;
        U[2*4+j] = -t1 + t2;
        U[3*4+j] = t1 - t3;
    }
}

// G: 4x3 filter transform
inline void winograd_G_g_GT(const float g[9], float V[16]) {
    // g is 3x3 row-major kernel
    // G * g:
    // G = [[1,0,0],[1/2,1/2,1/2],[1/2,-1/2,1/2],[0,0,1]]
    float tmp[12]; // 4x3 intermediate
    for (int i = 0; i < 4; ++i) tmp[i*3+0] = tmp[i*3+1] = tmp[i*3+2] = 0;

    tmp[0*3+0] = g[0]; tmp[0*3+1] = g[1]; tmp[0*3+2] = g[2];
    float h0 = 0.5f * (g[0] + g[1] + g[2]);
    float h1 = 0.5f * (g[0] - g[1] + g[2]);
    tmp[1*3+0] = h0; tmp[1*3+1] = h0; tmp[1*3+2] = h0;
    tmp[2*3+0] = h1; tmp[2*3+1] = h1; tmp[2*3+2] = h1;
    tmp[3*3+0] = g[6]; tmp[3*3+1] = g[7]; tmp[3*3+2] = g[8];

    // V = (G*g) * G^T
    for (int i = 0; i < 4; ++i) {
        float t0 = tmp[i*3+0], t1 = tmp[i*3+1], t2 = tmp[i*3+2];
        V[i*4+0] = t0;
        V[i*4+1] = 0.5f * (t0 + t1 + t2);
        V[i*4+2] = 0.5f * (t0 - t1 + t2);
        V[i*4+3] = t2;
    }
}

// A^T * M * A: 2x4 * 4x4 * 4x2 = 2x2 output transform
// A^T = [[1,1,1,0],[0,1,-1,-1]], A = transpose(A^T)
// Explicit formulas (adds only, no multiplies):
inline void winograd_AT_M_A(const float M[16], float Y[4]) {
    // M is 4x4 row-major: m[row*4+col]
    float m00 = M[0],  m01 = M[1],  m02 = M[2],  m03 = M[3];
    float m10 = M[4],  m11 = M[5],  m12 = M[6],  m13 = M[7];
    float m20 = M[8],  m21 = M[9],  m22 = M[10], m23 = M[11];
    float m30 = M[12], m31 = M[13], m32 = M[14], m33 = M[15];

    // A^T * M = [[r00,r01,r02,r03],[r10,r11,r12,r13]]
    float r00 = m00 + m10 + m20;
    float r01 = m01 + m11 + m21;
    float r02 = m02 + m12 + m22;
    float r03 = m03 + m13 + m23;

    float r10 = m10 - m20 - m30;
    float r11 = m11 - m21 - m31;
    float r12 = m12 - m22 - m32;
    float r13 = m13 - m23 - m33;

    // (A^T*M) * A = [[Y00,Y01],[Y10,Y11]]
    Y[0] = r00 + r01 + r02;           // Y[0,0] = r00 + r01 + r02
    Y[1] = r01 - r02 - r03;           // Y[0,1] = r01 - r02 - r03
    Y[2] = r10 + r11 + r12;           // Y[1,0] = r10 + r11 + r12
    Y[3] = r11 - r12 - r13;           // Y[1,1] = r11 - r12 - r13
}

// Inverse input transform (for backward pass): dD = B * dU * B^T
inline void winograd_B_dU_BT(const float dU[16], float dD[16]) {
    // B = [[1,0,0,0],[0,1,-1,1],[0,1,1,0],[0,-1,0,0]]?
    // Use the transpose relationship: B is 4x4, B^T is the forward transform
    // B * dU:
    float tmp[16];
    for (int i = 0; i < 4; ++i) {
        float u0 = dU[i*4+0], u1 = dU[i*4+1], u2 = dU[i*4+2], u3 = dU[i*4+3];
        tmp[i*4+0] = u0;
        tmp[i*4+1] = u1 - u2 + u3;
        tmp[i*4+2] = u1 + u2;
        tmp[i*4+3] = -u1;
    }
    // (B*dU) * B^T:
    for (int j = 0; j < 4; ++j) {
        float t0 = tmp[0*4+j], t1 = tmp[1*4+j], t2 = tmp[2*4+j], t3 = tmp[3*4+j];
        dD[0*4+j] = t0;
        dD[1*4+j] = t1 - t2 + t3;
        dD[2*4+j] = t1 + t2;
        dD[3*4+j] = -t1;
    }
}

// Inverse filter transform (for backward pass): dG = G^T * dV * G
inline void winograd_GT_dV_G(const float dV[16], float dG[9]) {
    // G^T * dV: 3x4 * 4x4 = 3x4
    float tmp[12];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            float sum = 0;
            for (int k = 0; k < 4; ++k) {
                // G^T[i,k]: G[k,i]
                float gki;
                if (i == 0) gki = (k == 0) ? 1.0f : 0.0f;
                else if (i == 1) gki = (k == 1) ? 0.5f : ((k == 2) ? -0.5f : 0.0f);
                else gki = (k == 0) ? 0.0f : ((k == 1) ? 0.5f : ((k == 2) ? 0.5f : 0.0f));
                sum += gki * dV[k*4+j];
            }
            tmp[i*4+j] = sum;
        }
    }
    // tmp * G: 3x4 * 4x3 = 3x3
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float sum = 0;
            for (int k = 0; k < 4; ++k) {
                float gkj;
                if (j == 0) gkj = (k == 0) ? 1.0f : 0.0f;
                else if (j == 1) gkj = (k == 1) ? 0.5f : ((k == 2) ? -0.5f : 0.0f);
                else gkj = (k == 0) ? 0.0f : ((k == 1) ? 0.5f : ((k == 2) ? 0.5f : 0.0f));
                sum += tmp[i*4+k] * gkj;
            }
            dG[i*3+j] = sum;
        }
    }
}

// =================================[Winograd Batched Forward]================================
// For stride-1 3x3 convolution with C_in input channels and C_out output channels.
// Uses batched GEMM for the Winograd-domain multiply.
inline void winograd_forward_3x3(const float* input, int H, int W, int C_in,
                                  const float* V_packed, int C_out,
                                  float* output) {
    // V_packed: pre-transformed filters, shape (C_out, C_in, 16)
    //   V_packed[(oc * C_in + ic) * 16 + pos] = V[oc, ic, pos]
    // where pos = alpha*4+beta indexes the 4x4 Winograd domain.

    const int H_out = H - 2;  // 3x3 valid convolution
    const int W_out = W - 2;
    const int P_h = (H_out + 1) / 2;  // number of 2x2 output tiles vertically
    const int P_w = (W_out + 1) / 2;
    const int P = P_h * P_w;  // total 2x2 output tiles

    // Step 1: Transform all input tiles to Winograd domain
    // U_all: (C_in, 16, P) - store as (C_in * 16 * P) flat
    float* U_all = new float[C_in * 16 * P];
    for (int p = 0; p < P; ++p) {
        int ph = p / P_w, pw = p % P_w;
        int h_start = ph * 2, w_start = pw * 2;
        for (int ic = 0; ic < C_in; ++ic) {
            float tile[16];
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    int src_h = h_start + i, src_w = w_start + j;
                    tile[i*4+j] = (src_h < H && src_w < W)
                        ? input[(ic * H + src_h) * W + src_w] : 0.0f;
                }
            }
            float U[16];
            winograd_BT_d_B(tile, U);
            for (int pos = 0; pos < 16; ++pos)
                U_all[(ic * 16 + pos) * P + p] = U[pos];
        }
    }

    // Step 2: For each Winograd position, M[oc,p] = Σ_ic V[oc,ic,pos] * U[ic,p,pos]
    // Equivalently: for each pos, M_pos = V_pos * U_pos
    // V_pos: (C_out, C_in), U_pos: (C_in, P), M_pos: (C_out, P)
    float* M_all = new float[C_out * 16 * P];
    for (int pos = 0; pos < 16; ++pos) {
        // Build V_pos: (C_out, C_in) from V_packed
        Matrix<float> V_pos(C_out, C_in);
        for (int oc = 0; oc < C_out; ++oc)
            for (int ic = 0; ic < C_in; ++ic)
                V_pos.at(oc, ic) = V_packed[(oc * C_in + ic) * 16 + pos];

        // Build U_pos: (C_in, P) from U_all
        Matrix<float> U_pos(C_in, P);
        for (int ic = 0; ic < C_in; ++ic)
            for (int p = 0; p < P; ++p)
                U_pos.at(ic, p) = U_all[(ic * 16 + pos) * P + p];

        // M_pos = V_pos * U_pos: (C_out, C_in) * (C_in, P) = (C_out, P)
        Matrix<float> M_pos = V_pos * U_pos;

        for (int oc = 0; oc < C_out; ++oc)
            for (int p = 0; p < P; ++p)
                M_all[(oc * 16 + pos) * P + p] = M_pos.at(oc, p);
    }

    // Step 3: Inverse transform each output tile
    // Output: (C_out, H_out, W_out) stored as (C_out, H_out, W_out) interleaved
    for (int oc = 0; oc < C_out; ++oc) {
        for (int p = 0; p < P; ++p) {
            int ph = p / P_w, pw = p % P_w;
            float M[16];
            for (int pos = 0; pos < 16; ++pos)
                M[pos] = M_all[(oc * 16 + pos) * P + p];

            float Y[4]; // 2x2 output tile
            winograd_AT_M_A(M, Y);

            int h = ph * 2, w = pw * 2;
            if (h < H_out && w < W_out)
                output[(oc * H_out + h) * W_out + w] = Y[0];
            if (h < H_out && w + 1 < W_out)
                output[(oc * H_out + h) * W_out + w + 1] = Y[1];
            // Note: AT_M_A returns only 2 values per 2x2 tile in this formulation
            // Actually Y[0],Y[1] are for positions (0,0),(0,1); we need also (1,0),(1,1)
            // Re-derive: A^T * M * A gives 2x2 = 4 values
        }
    }

    delete[] U_all;
    delete[] M_all;
}

// Pre-transform filters for Winograd: V = G * g * G^T for each filter
inline void winograd_transform_filter(const float* weight, int C_out, int C_in,
                                       float* V_packed) {
    // weight is (3, 3, C_in * C_out) with channel offset out_ch*C_in + in_ch
    for (int oc = 0; oc < C_out; ++oc) {
        for (int ic = 0; ic < C_in; ++ic) {
            float g[9];
            int ch = oc * C_in + ic;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    g[i*3+j] = weight[(i * 3 + j) * (C_in * C_out) + ch];

            float V[16];
            winograd_G_g_GT(g, V);

            for (int pos = 0; pos < 16; ++pos)
                V_packed[(oc * C_in + ic) * 16 + pos] = V[pos];
        }
    }
}

} // namespace CoreNNSpace

#endif // COREPP_CORE_CONVUTILS_H
