# CorePP — C++ Deep Learning Library

纯 C++20 深度学习库，支持 SIMD (AVX2/NEON) 加速和 CUDA GPU 推理。

## 目录

- [特性](#特性)
- [快速开始](#快速开始)
- [构建](#构建)
- [文档](#文档)
- [项目结构](#项目结构)

## 特性

- **20+ 层**: Linear, Conv2d, RNN, LSTM, GRU, MaxPool2d, BatchNorm, LayerNorm, Dropout, Embedding, GELU, Swish, Residual, ResNetBlock
- **矩阵运算**: GotoBLAS 风格 packed GEMM 微内核，AVX2 SIMD 加速
- **6 种优化器**: SGD, Momentum, Adam, AdamW, RMSProp, Adagrad
- **学习率调度**: Cosine Annealing, Step Decay
- **GPU 加速**: `model.cuda()` 一行搬运，全程 GPU 运算
- **梯度检验**: 数值梯度 vs 分析梯度对比

## 快速开始

```cpp
#include "CorePP.h"
using namespace CoreNNSpace;

int main() {
    auto fnn = nn::FNN({4, 8, 16, 8, 4}, nn::Sigmoid);

    Matrix<float> x(4, 1); x << 3 << 4 << 2 << 1;
    Matrix<float> y(4, 1); y << 1 << 0 << 0 << 0;

    nn::Trainer(fnn, nn::MSE, Optim(fnn.getParams(), Adam, 0.01f))
        .fit(x, y, 300, [](int e, float loss) {
            printf("epoch %d: %.6f\n", e, loss);
        });

    auto out = fnn.forward(x);
    out.Analysis("Prediction");
}
```

## 构建

### 静态库
```bash
cmake -G "MinGW Makefiles" -B _build
cmake --build _build
./_build/examples/example_fnn
```

### 动态库 (DLL)
```bash
cmake -G "MinGW Makefiles" -B _build_shared -DBUILD_SHARED_LIBS=ON
cmake --build _build_shared
g++ -std=c++20 -I CorePP examples/basic/main.cpp \
    -L_build_shared/lib -lCorePP -fopenmp -o example.exe
PATH="$PWD/_build_shared/bin:$PATH" ./example.exe
```

### CUDA
```bash
cmake -G "MinGW Makefiles" -B _build -DCOREPP_ENABLE_CUDA=ON
cmake --build _build
```

## 文档

| 文档 | 内容 |
|------|------|
| [快速入门](docs/QuickStart.md) | 安装、构建、第一个模型 |
| [API 参考](docs/API.md) | 所有类和函数 |
| [层说明](docs/Layers.md) | 全部 20+ 层详解 |
| [自定义模型](docs/CustomModel.md) | 5 分钟开发自己的层 |
| [CUDA 指南](docs/CUDA.md) | GPU 加速配置 |

## 项目结构

```
CorePP/
├── Core/           # 矩阵、GEMM、随机数
├── Layers/         # 神经网络层
├── Optimizers/     # 优化算法
├── Cuda/           # CUDA 后端
├── Utils/          # 工具函数
├── tests/          # 单元测试
├── examples/       # 使用示例
├── docs/           # 文档
├── cmake/          # CMake 模块
├── CorePP.h        # 主头文件
└── Model.h         # 高层 API
```
