# CUDA GPU 加速

## 环境要求

- **CUDA Toolkit** 11.0+
- **Visual Studio Build Tools** 2022 (提供 MSVC 编译器)
- **MinGW-w64** (主编译器)
- **CMake** 3.18+

## 工作原理

```
.cpp 文件 → MinGW g++ 编译
.cu 文件  → nvcc + MSVC cl.exe 编译
         → g++ 链接所有 .obj
```

CUDA 代码通过纯 C 接口 (`extern "C"`) 暴露，确保 MinGW 和 MSVC ABI 兼容。

## 一键构建

```bash
cmake -G "MinGW Makefiles" -B _build -DCOREPP_ENABLE_CUDA=ON
cmake --build _build
```

CMake 自动检测:
1. CUDA Toolkit
2. MSVC cl.exe 路径
3. 如任一缺失 → 自动回退 CPU 模式

## CUDA 内核

| 内核 | 文件 | 说明 |
|------|------|------|
| `gemm_kernel` | `cuda_gemm.cu` | 32×32 tiled GEMM，共享内存 |
| `sigmoid/tanh/relu` | `cuda_elementwise.cu` | 逐元素激活，256 threads |
| `rnn_cell` | `cuda_rnn.cu` | RNN 前向，batch 并行 |
| `lstm_cell` | `cuda_rnn.cu` | LSTM 4门融合，batch 并行 |

## 使用 CUDA

```cpp
#include "Core/CudaBridge.h"

// 运行时检测
if (CudaDevice::instance().available()) {
    // GPU 可用
    cuda_gemm(M, N, K, A, B, C);
    cuda_sigmoid(data, n);
}

// 或使用 dispatch (自动 CPU fallback)
CudaDevice::instance().dispatch(
    []{ /* CPU 实现 */ },
    []{ /* CUDA 实现 */ }
);
```

## 性能

CUDA 内核适合大批量数据。小矩阵 (< 4096 元素) 建议用 CPU GEMM 微内核。
