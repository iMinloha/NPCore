#ifndef COREPP_CORE_GEMM_H
#define COREPP_CORE_GEMM_H

// =================================[GotoBLAS-style GEMM Micro-Kernel]================================
//
// Reference: Goto & van de Geijn (2008) "Anatomy of High-Performance Matrix Multiplication"
//
// 6x16 AVX2 micro-kernel with packing.
//   - MR=6 rows of A, NR=16 cols of B — uses all 16 YMM registers
//   - Pack A into MR×K column-major panel (L1 cache)
//   - Pack B into K×NR row-major panel (L1 cache)
//   - Software prefetch for next B/A panels
//
// Register map (16 YMM, 256-bit):
//   C00..C51: 12 regs — accumulator for each (row, half-col) pair
//   B0, B1:    2 regs — current B[k] vector (2 × 8 floats)
//   A_bcast:   1 reg  — broadcast A[row][k]
//   (1 spare)

#ifdef __AVX__
#include <immintrin.h>

#define GEMM_MR 6
#define GEMM_NR 16

// Pack A panel: (MR, K) sub-block of A → A_packed[MR * K]
// A is row-major with leading dimension lda (= col).
// A_packed stores MR rows of length K, column-major: A_packed[row * K + k]
static inline void gemm_pack_A(int K, const float* A, int lda,
                                float* A_packed) {
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < GEMM_MR; ++i) {
            A_packed[i * K + k] = A[i * lda + k];
        }
    }
}

// Pack B panel: (K, NR) sub-block of B → B_packed[K * NR]
// B is row-major with leading dimension ldb (= B_col).
// B_packed stores NR cols, row-major: B_packed[k * NR + j]
static inline void gemm_pack_B(int K, const float* B, int ldb,
                                float* B_packed) {
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < GEMM_NR; ++j) {
            B_packed[k * GEMM_NR + j] = B[k * ldb + j];
        }
    }
}

// 6x16 micro-kernel: C[MR, NR] += A_packed[MR, K] * B_packed[K, NR]
// C is row-major with ldc. Updated in-place.
static inline void gemm_micro_kernel_6x16(int K,
                                           const float* A_packed,
                                           const float* B_packed,
                                           float* C, int ldc) {
    // Load C accumulators (12 YMM regs)
    __m256 C00 = _mm256_loadu_ps(&C[0 * ldc + 0]);
    __m256 C01 = _mm256_loadu_ps(&C[0 * ldc + 8]);
    __m256 C10 = _mm256_loadu_ps(&C[1 * ldc + 0]);
    __m256 C11 = _mm256_loadu_ps(&C[1 * ldc + 8]);
    __m256 C20 = _mm256_loadu_ps(&C[2 * ldc + 0]);
    __m256 C21 = _mm256_loadu_ps(&C[2 * ldc + 8]);
    __m256 C30 = _mm256_loadu_ps(&C[3 * ldc + 0]);
    __m256 C31 = _mm256_loadu_ps(&C[3 * ldc + 8]);
    __m256 C40 = _mm256_loadu_ps(&C[4 * ldc + 0]);
    __m256 C41 = _mm256_loadu_ps(&C[4 * ldc + 8]);
    __m256 C50 = _mm256_loadu_ps(&C[5 * ldc + 0]);
    __m256 C51 = _mm256_loadu_ps(&C[5 * ldc + 8]);

    // Prefetch first B panel
    _mm_prefetch((const char*)(B_packed + 0 * GEMM_NR), _MM_HINT_T0);

    for (int k = 0; k < K; ++k) {
        // Prefetch next B panel (ahead by 8 iterations = 1 cache line)
        _mm_prefetch((const char*)(B_packed + (k + 8) * GEMM_NR), _MM_HINT_T0);

        // Load B_packed[k][0..7] and [k][8..15]
        __m256 B0 = _mm256_loadu_ps(&B_packed[k * GEMM_NR + 0]);
        __m256 B1 = _mm256_loadu_ps(&B_packed[k * GEMM_NR + 8]);

        // Row 0
        __m256 A0 = _mm256_broadcast_ss(&A_packed[0 * K + k]);
        C00 = _mm256_fmadd_ps(A0, B0, C00);
        C01 = _mm256_fmadd_ps(A0, B1, C01);

        // Row 1
        __m256 A1 = _mm256_broadcast_ss(&A_packed[1 * K + k]);
        C10 = _mm256_fmadd_ps(A1, B0, C10);
        C11 = _mm256_fmadd_ps(A1, B1, C11);

        // Row 2
        __m256 A2 = _mm256_broadcast_ss(&A_packed[2 * K + k]);
        C20 = _mm256_fmadd_ps(A2, B0, C20);
        C21 = _mm256_fmadd_ps(A2, B1, C21);

        // Row 3
        __m256 A3 = _mm256_broadcast_ss(&A_packed[3 * K + k]);
        C30 = _mm256_fmadd_ps(A3, B0, C30);
        C31 = _mm256_fmadd_ps(A3, B1, C31);

        // Row 4
        __m256 A4 = _mm256_broadcast_ss(&A_packed[4 * K + k]);
        C40 = _mm256_fmadd_ps(A4, B0, C40);
        C41 = _mm256_fmadd_ps(A4, B1, C41);

        // Row 5
        __m256 A5 = _mm256_broadcast_ss(&A_packed[5 * K + k]);
        C50 = _mm256_fmadd_ps(A5, B0, C50);
        C51 = _mm256_fmadd_ps(A5, B1, C51);
    }

    // Store C accumulators back
    _mm256_storeu_ps(&C[0 * ldc + 0], C00);
    _mm256_storeu_ps(&C[0 * ldc + 8], C01);
    _mm256_storeu_ps(&C[1 * ldc + 0], C10);
    _mm256_storeu_ps(&C[1 * ldc + 8], C11);
    _mm256_storeu_ps(&C[2 * ldc + 0], C20);
    _mm256_storeu_ps(&C[2 * ldc + 8], C21);
    _mm256_storeu_ps(&C[3 * ldc + 0], C30);
    _mm256_storeu_ps(&C[3 * ldc + 8], C31);
    _mm256_storeu_ps(&C[4 * ldc + 0], C40);
    _mm256_storeu_ps(&C[4 * ldc + 8], C41);
    _mm256_storeu_ps(&C[5 * ldc + 0], C50);
    _mm256_storeu_ps(&C[5 * ldc + 8], C51);
}

// =================================[Main GEMM Entry Point]================================
// C[M, N] += A[M, K] * B[K, N]   (row-major, C updated in-place)
// Uses block-tiling + packing + 6x16 micro-kernel.
// Alignment: all pointers assumed at least 8-byte aligned.

static inline void gemm(int M, int N, int K,
                         const float* A, int lda,
                         const float* B, int ldb,
                         float* C, int ldc) {
    // Small-matrix fast path: skip packing overhead for tiny matmuls
    // Threshold: total FLOPs < ~4096 (e.g., 8x8x64 = 4096)
    if (M * N * K < 4096) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = C[i * ldc + j], comp = 0.0f;
                for (int k = 0; k < K; ++k) {
                    float y = A[i * lda + k] * B[k * ldb + j] - comp;
                    float t = sum + y;
                    comp = (t - sum) - y;
                    sum = t;
                }
                C[i * ldc + j] = sum;
            }
        }
        return;
    }

    // Block sizes: tuned for 32KB L1 cache
    // A_panel: MR × KC = 6 × 256 floats = 6KB
    // B_panel: KC × NR = 256 × 16 floats = 16KB
    // Total L1 ~22KB < 32KB ✓
    constexpr int KC = 256;
    constexpr int MC = 384;  // block rows of C (384 × 6-wide panels)
    constexpr int NC = 384;  // block cols of C

    // Allocate packing buffers on stack (aligned to 32 bytes)
    // Max sizes: A_packed[MC * KC] = 384*256 = 98K, B_packed[KC * NC] = 256*384 = 98K
    // For large matrices, use smaller dynamic allocation
    // Stack allocation for small/medium matrices:
    alignas(32) float A_packed[GEMM_MR * KC];   // 6 * 256 = 1536 floats = 6KB
    alignas(32) float B_packed[KC * GEMM_NR];   // 256 * 16 = 4096 floats = 16KB

    for (int jc = 0; jc < N; jc += NC) {
        int jc_end = (jc + NC < N) ? jc + NC : N;
        for (int pc = 0; pc < K; pc += KC) {
            int pc_end = (pc + KC < K) ? pc + KC : K;
            int Kc = pc_end - pc;

            for (int jr = jc; jr < jc_end; jr += GEMM_NR) {
                int nr = (jc_end - jr >= GEMM_NR) ? GEMM_NR : (jc_end - jr);

                // Pack B panel: B[pc:pc_end, jr:jr+nr]
                // Always pack to GEMM_NR width (pad with zeros for partial blocks)
                for (int k = 0; k < Kc; ++k) {
                    for (int j = 0; j < nr; ++j)
                        B_packed[k * GEMM_NR + j] = B[(pc + k) * ldb + (jr + j)];
                    for (int j = nr; j < GEMM_NR; ++j)
                        B_packed[k * GEMM_NR + j] = 0.0f;
                }

                for (int ic = 0; ic < M; ic += MC) {
                    int ic_end = (ic + MC < M) ? ic + MC : M;

                    for (int ir = ic; ir < ic_end; ir += GEMM_MR) {
                        int mr = (ic_end - ir >= GEMM_MR) ? GEMM_MR : (ic_end - ir);

                        // Full block: use micro-kernel. Edge block: scalar fallback.
                        if (mr == GEMM_MR && nr == GEMM_NR) {
                            // Pack A full panel
                            for (int k = 0; k < Kc; ++k)
                                for (int i = 0; i < GEMM_MR; ++i)
                                    A_packed[i * Kc + k] = A[(ir + i) * lda + (pc + k)];

                            gemm_micro_kernel_6x16(Kc, A_packed, B_packed,
                                                    &C[ir * ldc + jr], ldc);
                        } else {
                            // Scalar edge block with Kahan compensated summation
                            for (int ii = 0; ii < mr; ++ii) {
                                for (int jj = 0; jj < nr; ++jj) {
                                    float sum = C[(ir + ii) * ldc + (jr + jj)];
                                    float comp = 0.0f;  // Kahan compensation
                                    for (int kk = 0; kk < Kc; ++kk) {
                                        float y = A[(ir + ii) * lda + (pc + kk)]
                                                * B[(pc + kk) * ldb + (jr + jj)] - comp;
                                        float t = sum + y;
                                        comp = (t - sum) - y;
                                        sum = t;
                                    }
                                    C[(ir + ii) * ldc + (jr + jj)] = sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#else  // !__AVX__

// Fallback: scalar GEMM with basic blocking
static inline void gemm(int M, int N, int K,
                         const float* A, int lda,
                         const float* B, int ldb,
                         float* C, int ldc) {
    constexpr int BLOCK = 64;
    for (int i = 0; i < M; i += BLOCK) {
        for (int j = 0; j < N; j += BLOCK) {
            for (int k = 0; k < K; k += BLOCK) {
                int ie = (i + BLOCK < M) ? i + BLOCK : M;
                int je = (j + BLOCK < N) ? j + BLOCK : N;
                int ke = (k + BLOCK < K) ? k + BLOCK : K;
                for (int ii = i; ii < ie; ++ii) {
                    for (int jj = j; jj < je; ++jj) {
                        float sum = C[ii * ldc + jj];
                        for (int kk = k; kk < ke; ++kk)
                            sum += A[ii * lda + kk] * B[kk * ldb + jj];
                        C[ii * ldc + jj] = sum;
                    }
                }
            }
        }
    }
}

#endif  // __AVX__

#endif // COREPP_CORE_GEMM_H
