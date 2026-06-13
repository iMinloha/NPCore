# CorePP — C++ Deep Learning Library

纯 C++20 深度学习库，AVX2 SIMD + CUDA GPU 双后端加速。

![Architecture](assets/img/architecture.png)

## 目录

- [特性](#特性)
- [快速开始](#快速开始)
- [安装与构建](#安装与构建)
- [示例程序](#示例程序)
- [文档](#文档)
- [目录结构](#目录结构)

## 特性

**25+ 层**
`Linear` `Conv2d` `ConvTranspose2d` `MaxPool2d` `AvgPool2d` `AdaptiveAvgPool2d` `Flatten` `RNN` `LSTM` `GRU` `BatchNorm1d` `BatchNorm2d` `LayerNorm` `GroupNorm` `Dropout` `Embedding` `MultiHeadAttention` `Residual` `ResNetBlock`

**12 种激活**
`ReLU` `LeakyReLU` `Sigmoid` `Tanh` `SoftMax` `ELU` `SELU` `Softplus` `Mish` `GELU` `Swish`

**6 种损失**
`MSE` `MAE(L1)` `SmoothL1(Huber)` `CrossEntropy` `BinaryCrossEntropy` `KL Divergence`

**9 种优化器**
`SGD` `Momentum` `Adam` `AdamW` `RMSProp` `Adagrad` `Adadelta` `NAdam` `RAdam`

**训练工具**
`GradientClipping` (clip_by_norm / clip_by_value) · `Cosine Annealing` · `Step Decay`

**GPU 加速**: `model.cuda()` 一行搬运，全程 GPU 运算

**矩阵运算**: GotoBLAS 风格 packed GEMM 微内核 (6×16 AVX2)，Kahan 补偿求和

**梯度检验**: `numerical_gradient()` + `gradcheck()` 自动对比

### 归一化层对比

| 方法 | 归一化维度 | 依赖 Batch | 典型用途 |
|------|-----------|-----------|---------|
| BatchNorm1d | (N,) per feature | 是 | MLP |
| BatchNorm2d | (H,W) per channel | 是 | CNN, ResNet |
| LayerNorm | (F,) per sample | 否 | Transformer |
| GroupNorm | (H,W,C/G) per group | 否 | Detection / GAN |

![Normalization](assets/img/normalization.png)

### Transformer 支持

![MultiHeadAttention](assets/img/mha.png)

### 卷积上采样

![ConvTranspose2d](assets/img/convtranspose2d.png)

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
./_build/examples/test_fnn
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

14 个分类示例，覆盖基础层到 Transformer:

```bash
# 单元测试
./_build/examples/test_gradcheck       # 数值梯度 vs 解析梯度

# 基础层
./_build/examples/test_fnn             # 全连接网络: 4→8→16→8→4

# 卷积与池化
./_build/examples/test_cnn             # Conv→ReLU→Pool→Flatten→Linear
./_build/examples/test_convtranspose2d # 转置卷积 (U-Net 解码器)
./_build/examples/test_pooling         # AvgPool2d + AdaptiveAvgPool2d

# 循环网络
./_build/examples/test_rnn             # Elman RNN
./_build/examples/test_lstm            # 长短期记忆
./_build/examples/test_gru             # 门控循环单元

# 归一化
./_build/examples/test_batchnorm2d     # 2D 批归一化 (CNN)
./_build/examples/test_groupnorm       # 分组归一化 (检测/分割)

# 注意力
./_build/examples/test_mha             # 多头注意力 (Transformer)

# 优化器工具
./_build/examples/test_gradclip        # 梯度裁剪 (RNN/Transformer 训练)

# 架构组件
./_build/examples/test_resnet          # ResNet: Conv→ResBlock→Pool→Linear

# 数据加载
./_build/examples/test_dataloader      # 自定义数据集 + Train/Test 划分
```

> 详细说明见 [示例程序指南](docs/Examples.md)

## 文档

| 文档 | 内容 |
|------|------|
| [快速入门](docs/QuickStart.md) | 安装构建 + 各模型示例 (带注释) |
| [API 参考](docs/API.md) | 完整 API (每个参数类型和作用有注释) |
| [层说明](docs/Layers.md) | 全部 25+ 层数学公式与说明 |
| [开发指南](docs/DevGuide.md) | 实现原理 + 编码规范 + 自定义激活/损失/层模板 |
| [自定义模型](docs/CustomModel.md) | 继承 Module 开发自己的层 (简单/复合) |
| [示例程序](docs/Examples.md) | 14 个示例的教学文档 |
| [CUDA 指南](docs/CUDA.md) | GPU 编译与使用 |

## 目录结构

```
CorePP/
├── Core/                  # 矩阵库核心
│   ├── Matrix.hpp/.inl    #   泛型矩阵类 (2D/3D, SIMD运算符)
│   ├── GEMM.h             #   GotoBLAS 微内核 (6×16 AVX2, Kahan求和)
│   ├── ConvUtils.h        #   im2col / col2im / Winograd
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
├── Layers/                # 神经网络层 (25+)
│   ├── Module.h/.cpp      #   基类 (forward/backward/getParams/CleanGard)
│   ├── ParamInit.h        #   参数初始化 (Xavier/He/Uniform/Gaussian/Orthogonal)
│   ├── Sequence.h/.cpp    #   顺序容器
│   ├── Basic/             #   基础层
│   │   ├── Linear         #   全连接
│   │   ├── Flatten        #   展平
│   │   ├── Embedding      #   词嵌入
│   │   └── Dropout        #   随机失活
│   ├── Conv/              #   卷积
│   │   ├── Conv2d         #   2D 卷积 (im2col+GEMM)
│   │   ├── ConvTranspose2d#   转置卷积 (上采样)
│   │   └── Pool           #   MaxPool2d / AvgPool2d / AdaptiveAvgPool2d
│   ├── Recurrent/         #   循环网络
│   │   ├── RNN            #   Elman RNN (BPTT)
│   │   ├── LSTM           #   长短期记忆 (4门合并GEMM)
│   │   └── GRU            #   门控循环单元
│   ├── Normalization/     #   归一化
│   │   ├── BatchNorm      #   BatchNorm1d / BatchNorm2d
│   │   ├── LayerNorm      #   层归一化
│   │   └── GroupNorm      #   分组归一化
│   ├── Attention/         #   注意力
│   │   └── MultiHeadAttention  # Transformer 核心
│   └── Architecture/      #   架构组件
│       ├── Residual       #   残差连接包装器
│       └── ResNetBlock    #   ResNet 基础块
│
├── Optimizers/            # 优化器 + 工具
│   ├── Optimizer.h/.cpp   #   枚举 + Optim类 + dispatch
│   ├── SGD / Momentum / Adam / AdamW
│   ├── RMSProp / Adagrad / Adadelta / NAdam / RAdam
│   ├── LRScheduler.h      #   CosineAnnealing / StepDecay
│   └── GradientClipping.h #   clip_by_norm / clip_by_value
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
├── examples/              # 示例程序 (14个, 9分类)
│   ├── unit/              #   单元测试
│   ├── basic/             #   基础层
│   ├── conv/              #   卷积+池化
│   ├── recurrent/         #   循环网络
│   ├── normalization/     #   归一化
│   ├── attention/         #   注意力
│   ├── optimizers/        #   优化器工具
│   ├── architecture/      #   架构组件
│   └── data/              #   数据加载
│
├── assets/                # 图片 & 生成脚本
│   ├── img/               #   架构图、MHA流程图等
│   └── scripts/           #   Python matplotlib 绘图脚本
│
└── docs/                  # 开发文档 (7篇)
    ├── QuickStart.md      #   快速入门
    ├── API.md             #   API 参考
    ├── Layers.md          #   层详解 (数学公式)
    ├── DevGuide.md        #   开发指南
    ├── CustomModel.md     #   自定义模型
    ├── Examples.md        #   示例程序教学
    └── CUDA.md            #   GPU 加速配置
```
