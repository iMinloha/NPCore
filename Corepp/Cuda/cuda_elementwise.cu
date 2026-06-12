#include <cuda_runtime.h>
#include <math.h>
#include "cuda_runtime.h"

#define BLOCK_SIZE 256

// =================================[Sigmoid]================================
__global__ void sigmoid_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        data[i] = 1.0f / (1.0f + expf(-data[i]));
}

extern "C" int cuda_corepp_sigmoid(float* data, int n) {
    if (!data || n <= 0) return -1;
    float* d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sigmoid_kernel<<<blocks, BLOCK_SIZE>>>(d_data, n);

    cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    return 0;
}

// =================================[Tanh]================================
__global__ void tanh_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        data[i] = tanhf(data[i]);
}

extern "C" int cuda_corepp_tanh(float* data, int n) {
    if (!data || n <= 0) return -1;
    float* d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    tanh_kernel<<<blocks, BLOCK_SIZE>>>(d_data, n);

    cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    return 0;
}

// =================================[ReLU]================================
__global__ void relu_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        data[i] = fmaxf(0.0f, data[i]);
}

extern "C" int cuda_corepp_relu(float* data, int n) {
    if (!data || n <= 0) return -1;
    float* d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_kernel<<<blocks, BLOCK_SIZE>>>(d_data, n);

    cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    return 0;
}

// =================================[Element-wise multiply: a *= b]================================
__global__ void mul_kernel(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        a[i] *= b[i];
}

extern "C" int cuda_corepp_mul(float* a, const float* b, int n) {
    if (!a || !b || n <= 0) return -1;
    float *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mul_kernel<<<blocks, BLOCK_SIZE>>>(d_a, d_b, n);

    cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b);
    return 0;
}

// =================================[Element-wise add: a += b]================================
__global__ void add_kernel(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        a[i] += b[i];
}

extern "C" int cuda_corepp_add(float* a, const float* b, int n) {
    if (!a || !b || n <= 0) return -1;
    float *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_kernel<<<blocks, BLOCK_SIZE>>>(d_a, d_b, n);

    cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b);
    return 0;
}
