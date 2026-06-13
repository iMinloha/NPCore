#include <cuda_runtime.h>
#include "cuda_runtime.h"

// =================================[Tiled GEMM Kernel]================================
// 32×32 thread block, each thread computes one C element via dot product over shared memory tiles.
// A: M×K, B: K×N, C: M×N  (all row-major, C += A*B)

#define TILE_SIZE 32

__global__ void gemm_kernel(int M, int N, int K,
                             const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int k = 0; k < K; k += TILE_SIZE) {
        // Cooperative load A tile
        if (row < M && (k + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + (k + threadIdx.x)];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Cooperative load B tile
        if ((k + threadIdx.y) < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Dot product within tile
        for (int kk = 0; kk < TILE_SIZE; ++kk)
            sum += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;  // Note: overwrites, not accumulates
}

extern "C" int cuda_corepp_gemm(int M, int N, int K,
                                 const float* A, const float* B, float* C) {
    if (!A || !B || !C) return -1;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);

    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
