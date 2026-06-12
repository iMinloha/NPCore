#ifndef COREPP_CORE_CUDABRIDGE_H
#define COREPP_CORE_CUDABRIDGE_H

// =================================[CorePP CUDA Bridge — 自动 GPU 加速]================================
//
// 所有 Matrix 运算和 Activation 通过此桥接自动分发到 CUDA。
// 在 Layers/ 中新建模型不需要任何 CUDA 代码 — 自动获得 GPU 加速。
//
// 原理:
//   Matrix::operator*  → 检测 CUDA → cuda_gemm() 或 CPU gemm()
//   Activation::forward → 检测 CUDA → cuda_sigmoid/tanh() 或 CPU

#ifdef COREPP_ENABLE_CUDA
extern "C" {
#include "cuda_runtime.h"
}
#else
// Stubs when CUDA not compiled
inline int  cuda_corepp_init(void)           { return -1; }
inline void cuda_corepp_shutdown(void)        {}
inline int  cuda_corepp_has_device(void)      { return 0; }
inline void* cuda_corepp_malloc(int)          { return nullptr; }
inline void  cuda_corepp_free(void*)          {}
inline int   cuda_corepp_gemm(int, int, int, const float*, const float*, float*) { return -1; }
inline int   cuda_corepp_sigmoid(float*, int) { return -1; }
inline int   cuda_corepp_tanh(float*, int)    { return -1; }
inline int   cuda_corepp_relu(float*, int)    { return -1; }
inline int   cuda_corepp_rnn_cell(int, int, int, const float*, const float*, const float*,
                                   const float*, const float*, float*) { return -1; }
inline int   cuda_corepp_lstm_cell(int, int, int, const float*, const float*,
                                    const float*, const float*, const float*,
                                    float*, float*) { return -1; }
#endif

namespace CoreNNSpace {

// =================================[CudaDevice — 单例]================================
class CudaDevice {
public:
    static CudaDevice& instance() {
        static CudaDevice dev;
        return dev;
    }
    bool available() const { return has_cuda_; }

private:
    bool has_cuda_;
    CudaDevice()          { has_cuda_ = (cuda_corepp_init() == 0); }
    ~CudaDevice()         { if (has_cuda_) cuda_corepp_shutdown(); }
    CudaDevice(const CudaDevice&) = delete;
};

// =================================[GPU Memory]================================
inline void* cuda_malloc_device(size_t bytes) {
    (void)bytes;
#ifdef COREPP_ENABLE_CUDA
    return cuda_corepp_malloc((int)bytes);
#endif
    return nullptr;
}

inline void cuda_free_device(void* p) {
    (void)p;
#ifdef COREPP_ENABLE_CUDA
    if (p) cuda_corepp_free(p);
#endif
}

// =================================[GPU-Resident GEMM]================================
inline bool cuda_gemm_device(int M, int N, int K,
                              const float* d_A, const float* d_B, float* d_C) {
    (void)M; (void)N; (void)K; (void)d_A; (void)d_B; (void)d_C;
#ifdef COREPP_ENABLE_CUDA
    if (!CudaDevice::instance().available()) return false;
    return cuda_corepp_gemm(M, N, K, d_A, d_B, d_C) == 0;
#endif
    return false;
}

// =================================[CPU GEMM dispatch (H2D→kernel→D2H)]================================
inline bool cuda_should_use(int total_ops) {
    return CudaDevice::instance().available() && total_ops > 4096;
}

inline bool cuda_gemm_dispatch(int M, int N, int K,
                                const float* A, const float* B, float* C) {
    if (!cuda_should_use(M * N * K)) return false;
    return cuda_corepp_gemm(M, N, K, A, B, C) == 0;
}

// =================================[Activation dispatch]================================
inline bool cuda_sigmoid_dispatch(float* data, int n) {
    if (!cuda_should_use(n)) return false;
    return cuda_corepp_sigmoid(data, n) == 0;
}
inline bool cuda_tanh_dispatch(float* data, int n) {
    if (!cuda_should_use(n)) return false;
    return cuda_corepp_tanh(data, n) == 0;
}
inline bool cuda_relu_dispatch(float* data, int n) {
    if (!cuda_should_use(n)) return false;
    return cuda_corepp_relu(data, n) == 0;
}

} // namespace CoreNNSpace

#endif
