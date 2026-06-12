#include <cuda_runtime.h>
#include "cuda_runtime.h"

static int g_device_count = 0;
static int g_initialized  = 0;

extern "C" int cuda_corepp_init(void) {
    if (g_initialized) return 0;
    cudaError_t err = cudaGetDeviceCount(&g_device_count);
    if (err != cudaSuccess || g_device_count == 0) {
        g_device_count = 0;
        return -1;
    }
    // Use device 0
    cudaSetDevice(0);
    g_initialized = 1;
    return 0;
}

extern "C" void cuda_corepp_shutdown(void) {
    cudaDeviceReset();
    g_initialized = 0;
    g_device_count = 0;
}

extern "C" int cuda_corepp_device_count(void) {
    return g_device_count;
}

extern "C" int cuda_corepp_has_device(void) {
    return g_initialized ? 1 : 0;
}

extern "C" void* cuda_corepp_malloc(int bytes) {
    void* ptr = nullptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
}

extern "C" void cuda_corepp_free(void* ptr) {
    cudaFree(ptr);
}

extern "C" void cuda_corepp_memcpy_h2d(void* dst, const void* src, int bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

extern "C" void cuda_corepp_memcpy_d2h(void* dst, const void* src, int bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}
