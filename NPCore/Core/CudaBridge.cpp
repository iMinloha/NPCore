#include "Core/CudaBridge.h"

// =================================[CUDA Stubs (when CUDA not compiled)]================================
#ifndef NPCORE_ENABLE_CUDA
extern "C" {

int  cuda_corepp_init(void)           { return -1; }
void cuda_corepp_shutdown(void)        {}
int  cuda_corepp_has_device(void)      { return 0; }
void* cuda_corepp_malloc(int)          { return nullptr; }
void  cuda_corepp_free(void*)          {}
void  cuda_corepp_memcpy_h2d(void*, const void*, int) {}
void  cuda_corepp_memcpy_d2h(void*, const void*, int) {}
int   cuda_corepp_gemm(int, int, int, const float*, const float*, float*) { return -1; }
int   cuda_corepp_sigmoid(float*, int) { return -1; }
int   cuda_corepp_tanh(float*, int)    { return -1; }
int   cuda_corepp_relu(float*, int)    { return -1; }
int   cuda_corepp_rnn_cell(int, int, int, const float*, const float*, const float*,
                           const float*, const float*, float*) { return -1; }
int   cuda_corepp_lstm_cell(int, int, int, const float*, const float*,
                             const float*, const float*, const float*, const float*,
                             float*, float*) { return -1; }

} // extern "C"
#endif

namespace NPCore {

// =================================[CudaDevice - 单例]================================
CudaDevice& CudaDevice::instance() {
    static CudaDevice dev;
    return dev;
}

bool CudaDevice::available() const { return has_cuda_; }

CudaDevice::CudaDevice() {
    has_cuda_ = (cuda_corepp_init() == 0);
}

CudaDevice::~CudaDevice() {
    if (has_cuda_) cuda_corepp_shutdown();
}

// =================================[GPU Memory]================================
void* cuda_malloc_device(size_t bytes) {
    (void)bytes;
#ifdef NPCORE_ENABLE_CUDA
    return cuda_corepp_malloc((int)bytes);
#endif
    return nullptr;
}

void cuda_free_device(void* p) {
    (void)p;
#ifdef NPCORE_ENABLE_CUDA
    if (p) cuda_corepp_free(p);
#endif
}

// =================================[GPU-Resident GEMM]================================
bool cuda_gemm_device(int M, int N, int K,
                      const float* d_A, const float* d_B, float* d_C) {
    (void)M; (void)N; (void)K; (void)d_A; (void)d_B; (void)d_C;
#ifdef NPCORE_ENABLE_CUDA
    if (!CudaDevice::instance().available()) return false;
    return cuda_corepp_gemm(M, N, K, d_A, d_B, d_C) == 0;
#endif
    return false;
}

// =================================[CPU GEMM dispatch (H2D->kernel->D2H)]================================
bool cuda_should_use(int total_ops) {
    return CudaDevice::instance().available() && total_ops > 4096;
}

bool cuda_gemm_dispatch(int M, int N, int K,
                        const float* A, const float* B, float* C) {
    if (!cuda_should_use(M * N * K)) return false;
    return cuda_corepp_gemm(M, N, K, A, B, C) == 0;
}

// =================================[Activation dispatch]================================
bool cuda_sigmoid_dispatch(float* data, int n) {
    if (!cuda_should_use(n)) return false;
    return cuda_corepp_sigmoid(data, n) == 0;
}

bool cuda_tanh_dispatch(float* data, int n) {
    if (!cuda_should_use(n)) return false;
    return cuda_corepp_tanh(data, n) == 0;
}

bool cuda_relu_dispatch(float* data, int n) {
    if (!cuda_should_use(n)) return false;
    return cuda_corepp_relu(data, n) == 0;
}

} // namespace NPCore
