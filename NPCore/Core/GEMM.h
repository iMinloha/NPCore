#ifndef NPCORE_CORE_GEMM_H
#define NPCORE_CORE_GEMM_H

#include "Core/Export.h"

// =================================[GotoBLAS-style GEMM Micro-Kernel]================================
//
// Reference: Goto & van de Geijn (2008) "Anatomy of High-Performance Matrix Multiplication"
//
// 6x16 AVX2 micro-kernel with packing.
// C[M, N] += A[M, K] * B[K, N]   (row-major, C updated in-place)

NPCORE_API void gemm(int M, int N, int K,
                     const float* A, int lda,
                     const float* B, int ldb,
                     float* C, int ldc);

#endif // NPCORE_CORE_GEMM_H
