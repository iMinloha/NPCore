<p align="center">
  <img src="assets/img/architecture.png" alt="CorePP Architecture" width="720">
</p>

---

<p align="center">
  <strong>CorePP</strong> — 纯 C++20 深度学习库，AVX2 SIMD 与 CUDA GPU 双后端加速
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-20-blue?logo=cplusplus" alt="C++20">
  <img src="https://img.shields.io/badge/SIMD-AVX2-ff69b4" alt="AVX2">
  <img src="https://img.shields.io/badge/GPU-CUDA-76b900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/build-CMake-064f8c?logo=cmake" alt="CMake">
  <img src="https://img.shields.io/badge/tests-14%2F14-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

---

## 特性总览

| 类别 | 数量 | 内容 |
|:-----|:----:|:-----|
| 层 | **25+** | `Linear` `Conv2d` `ConvTranspose2d` `MaxPool2d` `AvgPool2d` `AdaptiveAvgPool2d` `Flatten` `RNN` `LSTM` `GRU` `BatchNorm1d` `BatchNorm2d` `LayerNorm` `GroupNorm` `Dropout` `Embedding` `MultiHeadAttention` `Residual` `ResNetBlock` |
| 激活函数 | **12** | `ReLU` `LeakyReLU` `Sigmoid` `Tanh` `SoftMax` `ELU` `SELU` `Softplus` `Mish` `GELU` `Swish` |
| 损失函数 | **6** | `MSE` `MAE` `SmoothL1` `CrossEntropy` `BCE` `KL Divergence` |
| 优化器 | **10** | `SGD` `Momentum` `Adam` `AdamW` `RMSProp` `Adagrad` `Adadelta` `NAdam` `RAdam` + `CosineLR` `StepLR` |
| 工具 | — | `GradientClipping` · `DataLoader` · `numerical_gradient` · `gradcheck` |

### 核心架构图

<p align="center">
  <img src="assets/img/normalization.png" alt="Normalization" width="540"><br>
  <img src="assets/img/mha.png" alt="MultiHeadAttention" width="540"><br>
  <img src="assets/img/convtranspose2d.png" alt="ConvTranspose2d" width="540">
</p>

---

## 快速开始

```cpp
#include "CorePP.h"
using namespace CoreNNSpace;

int main() {
    // 一行创建网络: 4 → 8 → 16 → 8 → 4
    auto net = nn::FNN({4, 8, 16, 8, 4}, nn::Sigmoid);

    Matrix<float> x(4, 1); x << 3 << 4 << 2 << 1;
    Matrix<float> y(4, 1); y << 1 << 0 << 0 << 0;

    nn::Trainer(net, nn::MSE, Optim(net.getParams(), Adam, 0.01f))
        .fit(x, y, 300, [](int e, float loss) {
            printf("epoch %d: %.6f\n", e, loss);
        });

    net.forward(x).Analysis("Prediction");
}
```

<details>
<summary><b>GPU 加速 — 一行代码</b></summary>

```cpp
net.cuda();   // 整个网络搬到 GPU
net.cpu();    // 搬回 CPU
```

</details>

<details>
<summary><b>Transformer 解码器 — 因果注意力</b></summary>

```cpp
MultiHeadAttention mha(/*d_model=*/512, /*heads=*/8, /*causal=*/true);
auto output = mha.forward(embedding);    // (seq_len, 512) -> (seq_len, 512)
```

</details>

---

## 编译与安装

| 平台 | 编译器 | 命令 |
|:-----|:------|:-----|
| CPU | MinGW GCC 12+ / MSVC 2022 | `cmake -G "MinGW Makefiles" -B _build && cmake --build _build` |
| GPU (CUDA) | + CUDA Toolkit 11.0+ | `cmake -B _build -DCOREPP_ENABLE_CUDA=ON` |
| DLL | 同 CPU | `cmake -B _build -DBUILD_SHARED_LIBS=ON` |

> **依赖:** CMake 3.18+, C++20 编译器。可选: CUDA 11.0+ 用于 GPU。

---

## 示例程序

> 14 个分类示例，每个都会打印输入/输出/期望值的精度分析。

```bash
# 单元测试
./_build/examples/test_gradcheck         # 数值梯度 vs 解析梯度

# 基础层
./_build/examples/test_fnn               # 全连接网络: 4->8->16->8->4

# 卷积
./_build/examples/test_cnn               # Conv -> ReLU -> Pool -> Flatten -> Linear
./_build/examples/test_convtranspose2d   # 转置卷积 (U-Net 解码器)
./_build/examples/test_pooling           # AvgPool2d + AdaptiveAvgPool2d

# 循环网络
./_build/examples/test_rnn               # Elman RNN (BPTT)
./_build/examples/test_lstm              # LSTM (4门合并GEMM)
./_build/examples/test_gru               # GRU  (2门)

# 归一化
./_build/examples/test_batchnorm2d       # 空间批归一化 (CNN)
./_build/examples/test_groupnorm         # 分组归一化 (检测/分割/GAN)

# 注意力
./_build/examples/test_mha               # 多头注意力 (Transformer)

# 工具
./_build/examples/test_gradclip          # 梯度裁剪 (RNN/Transformer 稳定性)

# 架构
./_build/examples/test_resnet            # ResNet: Conv -> ResBlock -> Pool -> Linear

# 数据
./_build/examples/test_dataloader        # 自定义数据集 + Train/Test 划分
```

> 详见 [示例程序指南](docs/Examples.md)

---

## 文档

由浅入深:

| 文档 | 难度 | 说明 |
|:-----|:----:|:-----|
| [快速入门](docs/QuickStart.md) | ★ | 安装 + 第一个模型 |
| [示例程序](docs/Examples.md) | ★★ | 14 个示例及期望输出 |
| [数据加载](docs/DataLoader.md) | ★★ | 图像/序列/CSV/自定义加载 |
| [层说明](docs/Layers.md) | ★★★ | 25+ 层数学公式与推导 |
| [API 参考](docs/API.md) | ★★★ | 完整类与方法参考 |
| [开发指南](docs/DevGuide.md) | ★★★★ | 实现原理 + 编码规范 |
| [自定义模型](docs/CustomModel.md) | ★★★★ | 继承 Module 开发新层 |
| [CUDA 指南](docs/CUDA.md) | ★★★★★ | GPU 编译与内核 |

---

## 性能

| 算子 | 配置 | 耗时 | 吞吐 |
|:-----|:-----|:-----|:-----|
| GEMM | 1024×1024 | 268 ms | **8.0 GFLOPS** |
| Conv2d | 64×64×3 -> 16ch, k3 | 4.1 ms | — |
| FNN | 256->512->256, 100 ep | 583 ms | 5.8 ms/epoch |
| LSTM | 32->128, seq50, 20 ep | 1545 ms | 77 ms/epoch |

> 测试环境: MinGW GCC 15.2, Intel i7-12700H, AVX2, `-march=native`

---

## 项目结构

```
CorePP/
├── Core/                          矩阵引擎 (GEMM, SIMD, CUDA bridge)
├── Layers/
│   ├── Basic/                     Linear  Flatten  Embedding  Dropout
│   ├── Conv/                      Conv2d  ConvTranspose2d  Pool (Avg/Max/Adaptive)
│   ├── Recurrent/                 RNN  LSTM  GRU
│   ├── Normalization/             BatchNorm1d/2d  LayerNorm  GroupNorm
│   ├── Attention/                 MultiHeadAttention
│   └── Architecture/              Residual  ResNetBlock  Sequence
├── Optimizers/                    SGD  Momentum  Adam  AdamW  ...  GradientClipping
├── Activations/                   12 种激活函数
├── Losses/                        6 种损失函数
├── Cuda/                          CUDA 内核 (GEMM, element-wise, RNN)
├── examples/                      14 个分类测试
├── assets/                        架构图与生成脚本
├── docs/                          7 篇文档
├── Model.h                        高层 API (nn::FNN, nn::CNN, nn::Trainer)
├── Autograd.h                     梯度检验工具
└── CorePP.h                       单头文件引入
```

---

<p align="center">
  <sub>Made with C++20 · AVX2 SIMD · CUDA · OpenMP</sub>
</p>
