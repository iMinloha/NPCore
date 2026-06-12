#include <cuda_runtime.h>
#include <math.h>
#include "cuda_runtime.h"

#define BLOCK_SIZE 256

// =================================[RNN Cell: h = tanh(W_ih*x + W_hh*h_prev + b)]================================
// Each thread computes one hidden unit.
// batch: number of independent samples processed in parallel
__global__ void rnn_cell_kernel(int batch, int input_size, int hidden_size,
                                 const float* __restrict__ W_ih,
                                 const float* __restrict__ W_hh,
                                 const float* __restrict__ b,
                                 const float* __restrict__ x,
                                 const float* __restrict__ h_prev,
                                 float* __restrict__ h_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // hidden unit index
    int bs = blockIdx.y;  // batch index
    if (i >= hidden_size || bs >= batch) return;

    // Dot product: W_ih[i,:] * x[bs,:]
    float sum = b[i];
    for (int j = 0; j < input_size; ++j)
        sum += W_ih[i * input_size + j] * x[bs * input_size + j];
    // Dot product: W_hh[i,:] * h_prev[bs,:]
    for (int j = 0; j < hidden_size; ++j)
        sum += W_hh[i * hidden_size + j] * h_prev[bs * hidden_size + j];

    h_out[bs * hidden_size + i] = tanhf(sum);
}

extern "C" int cuda_corepp_rnn_cell(int batch, int input_size, int hidden_size,
                                     const float* W_ih, const float* W_hh, const float* b,
                                     const float* x, const float* h_prev,
                                     float* h_out) {
    int n_w = hidden_size * input_size;
    int n_wh = hidden_size * hidden_size;
    int n_b = hidden_size;
    int n_x = batch * input_size;
    int n_h = batch * hidden_size;

    float *d_W_ih, *d_W_hh, *d_b, *d_x, *d_h_prev, *d_h_out;
    cudaMalloc(&d_W_ih, n_w * sizeof(float));
    cudaMalloc(&d_W_hh, n_wh * sizeof(float));
    cudaMalloc(&d_b, n_b * sizeof(float));
    cudaMalloc(&d_x, n_x * sizeof(float));
    cudaMalloc(&d_h_prev, n_h * sizeof(float));
    cudaMalloc(&d_h_out, n_h * sizeof(float));

    cudaMemcpy(d_W_ih, W_ih, n_w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_hh, W_hh, n_wh * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n_b * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, n_x * sizeof(float), cudaMemcpyHostToDevice);
    if (h_prev)
        cudaMemcpy(d_h_prev, h_prev, n_h * sizeof(float), cudaMemcpyHostToDevice);
    else
        cudaMemset(d_h_prev, 0, n_h * sizeof(float));

    int blocks_h = (hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(blocks_h, batch);
    rnn_cell_kernel<<<grid, BLOCK_SIZE>>>(batch, input_size, hidden_size,
                                           d_W_ih, d_W_hh, d_b, d_x, d_h_prev, d_h_out);

    cudaMemcpy(h_out, d_h_out, n_h * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_W_ih); cudaFree(d_W_hh); cudaFree(d_b);
    cudaFree(d_x); cudaFree(d_h_prev); cudaFree(d_h_out);
    return 0;
}

// =================================[LSTM Cell: 4-gate computation]================================
// W: (4*hidden, input+hidden), b: (4*hidden, 1)
// Computes f,i,o,g gates → c_t, h_t
__global__ void lstm_cell_kernel(int batch, int input_size, int hidden_size,
                                  const float* __restrict__ W,
                                  const float* __restrict__ b,
                                  const float* __restrict__ x,
                                  const float* __restrict__ h_prev,
                                  const float* __restrict__ c_prev,
                                  float* __restrict__ h_out,
                                  float* __restrict__ c_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // hidden unit
    int bs = blockIdx.y;  // batch index
    if (i >= hidden_size || bs >= batch) return;

    int total_dim = input_size + hidden_size;

    // Compute 4 gate pre-activations
    float f_gate = b[i];
    float i_gate = b[hidden_size + i];
    float o_gate = b[2 * hidden_size + i];
    float g_gate = b[3 * hidden_size + i];

    // W * [x; h_prev] for each gate
    for (int j = 0; j < total_dim; ++j) {
        float xh_j = (j < input_size) ? x[bs * input_size + j]
                                      : h_prev[bs * hidden_size + (j - input_size)];
        f_gate += W[i * total_dim + j] * xh_j;
        i_gate += W[(hidden_size + i) * total_dim + j] * xh_j;
        o_gate += W[(2 * hidden_size + i) * total_dim + j] * xh_j;
        g_gate += W[(3 * hidden_size + i) * total_dim + j] * xh_j;
    }

    // Apply activations
    float f = 1.0f / (1.0f + expf(-f_gate));
    float in = 1.0f / (1.0f + expf(-i_gate));
    float o  = 1.0f / (1.0f + expf(-o_gate));
    float g  = tanhf(g_gate);

    float c_prev_val = c_prev[bs * hidden_size + i];
    float c_val = f * c_prev_val + in * g;

    c_out[bs * hidden_size + i] = c_val;
    h_out[bs * hidden_size + i] = o * tanhf(c_val);
}

extern "C" int cuda_corepp_lstm_cell(int batch, int input_size, int hidden_size,
                                      const float* W, const float* b,
                                      const float* x, const float* h_prev, const float* c_prev,
                                      float* h_out, float* c_out) {
    int total_dim = input_size + hidden_size;
    int n_W = 4 * hidden_size * total_dim;
    int n_b = 4 * hidden_size;
    int n_x = batch * input_size;
    int n_h = batch * hidden_size;

    float *d_W, *d_b, *d_x, *d_h_prev, *d_c_prev, *d_h_out, *d_c_out;
    cudaMalloc(&d_W, n_W * sizeof(float));
    cudaMalloc(&d_b, n_b * sizeof(float));
    cudaMalloc(&d_x, n_x * sizeof(float));
    cudaMalloc(&d_h_prev, n_h * sizeof(float));
    cudaMalloc(&d_c_prev, n_h * sizeof(float));
    cudaMalloc(&d_h_out, n_h * sizeof(float));
    cudaMalloc(&d_c_out, n_h * sizeof(float));

    cudaMemcpy(d_W, W, n_W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n_b * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, n_x * sizeof(float), cudaMemcpyHostToDevice);
    if (h_prev) cudaMemcpy(d_h_prev, h_prev, n_h * sizeof(float), cudaMemcpyHostToDevice);
    else        cudaMemset(d_h_prev, 0, n_h * sizeof(float));
    if (c_prev) cudaMemcpy(d_c_prev, c_prev, n_h * sizeof(float), cudaMemcpyHostToDevice);
    else        cudaMemset(d_c_prev, 0, n_h * sizeof(float));

    int blocks_h = (hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(blocks_h, batch);
    lstm_cell_kernel<<<grid, BLOCK_SIZE>>>(batch, input_size, hidden_size,
                                            d_W, d_b, d_x, d_h_prev, d_c_prev,
                                            d_h_out, d_c_out);

    cudaMemcpy(h_out, d_h_out, n_h * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(c_out, d_c_out, n_h * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_W); cudaFree(d_b); cudaFree(d_x);
    cudaFree(d_h_prev); cudaFree(d_c_prev); cudaFree(d_h_out); cudaFree(d_c_out);
    return 0;
}
