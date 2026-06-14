#ifndef NPCORE_CORE_CUDABRIDGE_H
#define NPCORE_CORE_CUDABRIDGE_H

// =================================[NPCore CUDA Bridge - 自动 GPU 加速]================================
//
// 所有 Matrix 运算和 Activation 通过此桥接自动分发到 CUDA。
// 在 Layers/ 中新建模型不需要任何 CUDA 代码 - 自动获得 GPU 加速。
//
// 原理:
//   Matrix::operator*  -> 检测 CUDA -> cuda_gemm() 或 CPU gemm()
//   Activation::forward -> 检测 CUDA -> cuda_sigmoid/tanh() 或 CPU

#ifdef NPCORE_ENABLE_CUDA
extern "C" {
#include "cuda_runtime.h"
}
#else
// Declarations for when CUDA is not compiled (stubs defined in CudaBridge.cpp)
extern "C" {
int  cuda_corepp_init(void);
void cuda_corepp_shutdown(void);
int  cuda_corepp_has_device(void);
void* cuda_corepp_malloc(int bytes);
void  cuda_corepp_free(void* ptr);
void  cuda_corepp_memcpy_h2d(void* dst, const void* src, int bytes);
void  cuda_corepp_memcpy_d2h(void* dst, const void* src, int bytes);
int   cuda_corepp_gemm(int M, int N, int K, const float* A, const float* B, float* C);
int   cuda_corepp_sigmoid(float* data, int n);
int   cuda_corepp_tanh(float* data, int n);
int   cuda_corepp_relu(float* data, int n);
int   cuda_corepp_rnn_cell(int batch, int input_size, int hidden_size,
                           const float* x, const float* hx, const float* W_ih,
                           const float* W_hh, const float* b, float* hy);
int   cuda_corepp_lstm_cell(int batch, int input_size, int hidden_size,
                             const float* x, const float* hx, const float* cx,
                             const float* W_ih, const float* W_hh, const float* b,
                             float* hy, float* cy);
}
#endif

namespace NPCore {

// =================================[CudaDevice - 单例]================================
class CudaDevice {
public:
    static CudaDevice& instance();
    bool available() const;

private:
    bool has_cuda_;
    CudaDevice();
    ~CudaDevice();
    CudaDevice(const CudaDevice&) = delete;
};

// =================================[GPU Memory]================================
void* cuda_malloc_device(size_t bytes);
void  cuda_free_device(void* p);

// =================================[GPU-Resident GEMM]================================
bool cuda_gemm_device(int M, int N, int K,
                      const float* d_A, const float* d_B, float* d_C);

// =================================[CPU GEMM dispatch (H2D->kernel->D2H)]================================
bool cuda_should_use(int total_ops);
bool cuda_gemm_dispatch(int M, int N, int K,
                        const float* A, const float* B, float* C);

// =================================[Activation dispatch]================================
bool cuda_sigmoid_dispatch(float* data, int n);
bool cuda_tanh_dispatch(float* data, int n);
bool cuda_relu_dispatch(float* data, int n);

} // namespace NPCore

#endif
