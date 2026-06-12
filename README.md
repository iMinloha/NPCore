# CorePP — C++ Deep Learning Library

纯 C++20 深度学习库，AVX2 SIMD + CUDA GPU 双后端加速。

## 目录

- [性能](#性能)
- [特性](#特性)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [安装与构建](#安装与构建)
- [示例程序](#示例程序)
- [文档](#文档)

## 性能

| 算子 | 配置 | 耗时 | 吞吐 |
|------|------|------|------|
| GEMM | 1024×1024 | 268 ms | **8.0 GFLOPS** |
| Conv2d | 64×64×3 → 16ch, k3 | 4.1 ms | — |
| FNN | 256→512→256, 100 epochs | 583 ms | 5.8 ms/epoch |
| LSTM | 32→128, seq50, 20 epochs | 1545 ms | 77 ms/epoch |

> 测试环境: MinGW GCC 15.2, Intel i7-12700H, AVX2, -march=native

## 特性

**20+ 层**
`Linear` `Conv2d` `MaxPool2d` `Flatten` `RNN` `LSTM` `GRU` `BatchNorm` `LayerNorm` `Dropout` `Embedding` `Residual` `ResNetBlock`

**12 种激活**
`ReLU` `LeakyReLU` `Sigmoid` `Tanh` `SoftMax` `ELU` `SELU` `Softplus` `Mish` `GELU` `Swish`

**6 种损失**
`MSE` `MAE(L1)` `SmoothL1(Huber)` `CrossEntropy` `BinaryCrossEntropy` `KL Divergence`

**9 种优化器**
`SGD` `Momentum` `Adam` `AdamW` `RMSProp` `Adagrad` `Adadelta` `NAdam` `RAdam`

**学习率调度**
`Cosine Annealing` `Step Decay`

**GPU 加速**: `model.cuda()` 一行搬运，全程 GPU 运算

**矩阵运算**: GotoBLAS 风格 packed GEMM 微内核 (6×16 AVX2)，Kahan 补偿求和

**梯度检验**: `numerical_gradient()` + `gradcheck()` 自动对比

## 目录结构

```
CorePP/
├── Core/                  # 矩阵库核心
│   ├── Matrix.hpp/.inl    #   泛型矩阵类 (2D/3D, SIMD运算符)
│   ├── GEMM.h             #   GotoBLAS 微内核 (6×16 AVX2, Kahan求和)
│   ├── ConvUtils.h        #   im2col / col2im
│   ├── CudaBridge.h       #   CUDA 运行时检测 + 自动分发
│   ├── RandomGenerator    #   随机数生成 (均匀/高斯, 线程安全)
│   ├── LinearAlgebra.h    #   Gram-Schmidt 正交化
│   └── Assert.h / Size.h  #   断言 / 尺寸类型
│
├── Activations/           # 激活函数 (12种)
│   └── Activation.h/.cpp  #   ReLU LeakyReLU Sigmoid Tanh SoftMax
│                          #   ELU SELU Softplus Mish GELU Swish
├── Losses/                # 损失函数 (6种)
│   └── Loss.h             #   MSE MAE SmoothL1 CrossEntropy BCE KL
│
├── Layers/                # 神经网络层
│   ├── Module.h/.cpp      #   基类 (forward/backward/getParams/CleanGard)
│   ├── ParamInit.h        #   参数初始化 (Xavier/He/Uniform/Gaussian...)
│   ├── Sequence.h/.cpp    #   顺序容器
│   ├── Basic/             #   基础层
│   │   ├── Linear         #   全连接
│   │   ├── Flatten        #   展平
│   │   ├── Embedding      #   词嵌入
│   │   └── Dropout        #   随机失活
│   ├── Conv/              #   卷积
│   │   ├── Conv2d         #   2D 卷积 (im2col+GEMM)
│   │   └── Pool           #   MaxPool2d
│   ├── Recurrent/         #   循环网络
│   │   ├── RNN            #   Elman RNN (BPTT)
│   │   ├── LSTM           #   长短期记忆 (4门合并GEMM)
│   │   └── GRU            #   门控循环单元
│   ├── Normalization/     #   归一化
│   │   ├── BatchNorm      #   批归一化
│   │   └── LayerNorm      #   层归一化
│   └── Architecture/      #   架构组件
│       ├── Residual       #   残差连接包装器
│       └── ResNetBlock    #   ResNet 基础块
│
├── Optimizers/            # 优化器 (9种 + 调度器)
│   ├── Optimizer.h/.cpp   #   枚举 + Optim类 + dispatch
│   ├── SGD / Momentum / Adam / AdamW
│   ├── RMSProp / Adagrad / Adadelta
│   ├── NAdam / RAdam
│   └── LRScheduler.h      #   CosineAnnealing / StepDecay
│
├── Cuda/                  # CUDA 后端
│   ├── cuda_runtime.h     #   纯C API (MinGW兼容)
│   ├── cuda_gemm.cu       #   32×32 tiled GEMM
│   ├── cuda_elementwise.cu#   sigmoid/tanh/relu kernels
│   ├── cuda_rnn.cu        #   RNN/LSTM cell kernels
│   └── cuda_device.cu     #   设备初始化/内存管理
│
├── Utils/                 # 工具
│   └── Timer.h/.cpp       #   计时器
│
├── CorePP.h               # 主头文件 (包含所有模块)
├── Core.h / NN.h / Optim.h# 模块头文件
├── Model.h                # 高层API (FNN/CNN/Trainer)
├── Autograd.h             # 梯度检验工具
│
├── examples/              # 示例程序 (7个)
│   ├── example_fnn        #   全连接网络
│   ├── example_cnn        #   卷积 + 池化
│   ├── example_rnn        #   循环网络
│   ├── example_lstm       #   长短期记忆
│   ├── example_gru        #   门控循环
│   ├── example_resnet     #   残差网络
│   └── example_gradcheck  #   梯度检验
│
└── docs/                  # 开发文档
    ├── QuickStart.md       #   快速入门
    ├── API.md              #   API 参考
    ├── Layers.md           #   层详解
    ├── CustomModel.md      #   自定义模型指南
    └── CUDA.md             #   GPU 加速配置
```

## 快速开始

```cpp
#include "CorePP.h"
using namespace CoreNNSpace;

int main() {
    // 一行创建网络: 4→8→16→8→4
    auto net = nn::FNN({4, 8, 16, 8, 4}, nn::Sigmoid);

    Matrix<float> x(4, 1); x << 3 << 4 << 2 << 1;
    Matrix<float> y(4, 1); y << 1 << 0 << 0 << 0;

    // 训练
    nn::Trainer(net, nn::MSE, Optim(net.getParams(), Adam, 0.01f))
        .fit(x, y, 300, [](int e, float loss) {
            printf("epoch %d: %.6f\n", e, loss);
        });

    // 推理
    net.forward(x).Analysis("Prediction");
}
```

## 安装与构建

### 依赖
- MinGW-w64 (GCC 12+) 或 MSVC 2022
- CMake 3.18+
- 可选: CUDA Toolkit 11.0+ + VS Build Tools (GPU)

### CPU 编译
```bash
cmake -G "MinGW Makefiles" -B _build
cmake --build _build
./_build/examples/example_fnn
```

### GPU (CUDA) 编译
```bash
cmake -G "MinGW Makefiles" -B _build -DCOREPP_ENABLE_CUDA=ON
cmake --build _build
```

### 动态库 (DLL)
```bash
cmake -G "MinGW Makefiles" -B _build -DBUILD_SHARED_LIBS=ON
cmake --build _build
```

## 示例程序

```bash
# 运行所有示例
./_build/examples/example_gradcheck   # 梯度正确性验证
./_build/examples/example_fnn         # 全连接网络: 4→8→16→8→4
./_build/examples/example_cnn         # 卷积+池化: Conv→ReLU→Pool→Flatten→Linear
./_build/examples/example_rnn         # 循环网络: 序列记忆
./_build/examples/example_lstm        # LSTM: 长期记忆
./_build/examples/example_gru         # GRU: 序列记忆
./_build/examples/example_resnet      # ResNet: Conv→ResBlock→Pool→Linear
```

## 文档

| 文档 | 内容 |
|------|------|
| [快速入门](docs/QuickStart.md) | 安装构建 + 各模型示例 |
| [API 参考](docs/API.md) | 完整 API (Matrix/层/损失/优化器) |
| [开发指南](docs/DevGuide.md) | 实现原理 + 编码规范 + 二次开发 |
| [层说明](docs/Layers.md) | 全部 20+ 层数学公式 |
| [自定义模型](docs/CustomModel.md) | 继承 Module 开发自己的层 |
| [CUDA 指南](docs/CUDA.md) | GPU 编译与使用 |
