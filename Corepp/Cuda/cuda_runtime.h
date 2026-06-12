#ifndef CUDA_RUNTIME_H
#define CUDA_RUNTIME_H

// =================================[CorePP CUDA C-API]================================
// Pure C interface — safe to link from MinGW or any C/C++ compiler.
// All functions return 0 on success, non-zero on error.

#ifdef __cplusplus
extern "C" {
#endif

// --- Device management ---
int  cuda_corepp_init(void);                    // Initialize CUDA, returns 0 if GPU available
void cuda_corepp_shutdown(void);                // Release CUDA resources
int  cuda_corepp_device_count(void);            // Number of CUDA devices
int  cuda_corepp_has_device(void);              // 1 if CUDA available, 0 otherwise

// --- Memory management ---
void* cuda_corepp_malloc(int bytes);            // Allocate GPU memory
void  cuda_corepp_free(void* ptr);              // Free GPU memory
void  cuda_corepp_memcpy_h2d(void* dst, const void* src, int bytes);  // Host→Device
void  cuda_corepp_memcpy_d2h(void* dst, const void* src, int bytes);  // Device→Host

// --- GEMM: C[M,N] = A[M,K] * B[K,N] ---
int cuda_corepp_gemm(int M, int N, int K,
                      const float* A, const float* B, float* C);

// --- Element-wise operations (in-place) ---
int cuda_corepp_sigmoid(float* data, int n);     // data[i] = 1/(1+exp(-data[i]))
int cuda_corepp_tanh(float* data, int n);        // data[i] = tanh(data[i])
int cuda_corepp_relu(float* data, int n);        // data[i] = max(0, data[i])
int cuda_corepp_mul(float* a, const float* b, int n);  // a[i] *= b[i]
int cuda_corepp_add(float* a, const float* b, int n);  // a[i] += b[i]

// --- RNN cell fused: h = tanh(W_ih*x + W_hh*h_prev + b) ---
int cuda_corepp_rnn_cell(int batch, int input_size, int hidden_size,
                          const float* W_ih, const float* W_hh, const float* b,
                          const float* x, const float* h_prev,
                          float* h_out);

// --- LSTM cell fused: 4 gates in one GEMM ---
int cuda_corepp_lstm_cell(int batch, int input_size, int hidden_size,
                           const float* W, const float* b,
                           const float* x, const float* h_prev, const float* c_prev,
                           float* h_out, float* c_out);

#ifdef __cplusplus
}
#endif

#endif // CUDA_RUNTIME_H
