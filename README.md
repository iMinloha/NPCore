# NPCore — C++20 Deep Learning Library

**纯 C++20 深度学习库** · AVX2 SIMD + CUDA GPU 双后端 · 零外部依赖 · ONNX 互操作

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-20-blue?logo=cplusplus" alt="C++20">
  <img src="https://img.shields.io/badge/SIMD-AVX2-ff69b4" alt="AVX2">
  <img src="https://img.shields.io/badge/GPU-CUDA-76b900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/ONNX-import%2Fexport-005ced?logo=onnx" alt="ONNX">
  <img src="https://img.shields.io/badge/build-CMake-064f8c?logo=cmake" alt="CMake">
  <img src="https://img.shields.io/badge/tests-22%2B-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

---

## 特性

| 类别 | 数量 | 内容 |
|:-----|:----:|:-----|
| 层 | **30+** | `Linear` `Conv1d` `Conv2d` `ConvTranspose2d` `MaxPool2d` `AvgPool2d` `AdaptiveAvgPool2d` `Flatten` `RNN` `LSTM` `GRU` `BatchNorm1d` `BatchNorm2d` `LayerNorm` `GroupNorm` `InstanceNorm1d` `InstanceNorm2d` `Dropout` `Embedding` `MultiHeadAttention` `PositionalEncoding` `TransformerEncoder` `Residual` `ResNetBlock` `Sequence` `Concat` |
| 激活函数 | **11** | `ReLU` `LeakyReLU` `Sigmoid` `Tanh` `SoftMax` `ELU` `SELU` `Softplus` `Mish` `GELU` `Swish` |
| 损失函数 | **9** | `MSE` `MAE` `SmoothL1` `CrossEntropy` `BCE` `KL` + 所有对应梯度函数 |
| 优化器 | **10** | `SGD` `Momentum` `Adam` `AdamW` `RMSProp` `Adagrad` `Adadelta` `NAdam` `RAdam` + `CosineLR` `StepLR` |
| 互操作 | **ONNX** | `ONNXModel::load()` / `from_sequence()` / `save()` / `to_sequence()` |
| 工具 | — | `GradientClipping` · `DataLoader` · `numerical_gradient` · `gradcheck` · `save_model` · `load_model_weights` |

---

## 快速开始

```cpp
#include "NPCore.h"
#include "Model.h"
using namespace NPCore;
using namespace NPCore::nn;

int main() {
    // 一行建网: 4→8→16→8→4
    auto net = FNN({4, 8, 16, 8, 4}, Sigmoid);

    Matrix<float> x(4, 1); x << 3 << 4 << 2 << 1;
    Matrix<float> y(4, 1); y << 1 << 0 << 0 << 0;

    Trainer(net, MSE, Adam(0.01f))
        .fit(x, y, 300, [](int e, float loss) {
            printf("epoch %d: %.6f\n", e, loss);
        });

    net.forward(x).Analysis("Prediction");
}
```

<details>
<summary><b>ONNX 导出 — 与 PyTorch 互通</b></summary>

```cpp
// NPCore → ONNX: 导出模型供 Python/ONNX Runtime 使用
auto model = FNN({4, 8, 4}, ReLU);
auto onnx = ONNXModel::from_sequence(model, {1, 4}, "MLP");
onnx.save("model.onnx");

// ONNX → NPCore: 加载 PyTorch 导出的模型
auto loaded = ONNXModel::load("from_pytorch.onnx");
Sequence imported = loaded.to_sequence();  // 直接推理
imported.eval();
auto out = imported.forward(input);
```

详见 [ONNX 互操作文档](docs/ONNX.md)

</details>

<details>
<summary><b>嵌套模型 + 分支结构</b></summary>

```cpp
// Sequence 可嵌套
auto* encoder = new Sequence({new Linear(64,32), new ReLU(), new Linear(32,16)});
auto* decoder = new Sequence({new Linear(16,32), new ReLU(), new Linear(32,64)});
Sequence autoencoder({encoder, decoder});

// Concat 支持 Inception/U-Net 分支
auto* b1 = new Sequence({new Conv2d(1,4,3,1,1), new ReLU()});
auto* b2 = new Sequence({new Conv2d(1,3,5,1,2), new ReLU()});
Concat inception({b1, b2});
```

</details>

---

## 编译

```bash
git clone <repo> && cd Korea_C++
mkdir build && cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build . -j8
```

| 选项 | 默认 | 说明 |
|:-----|:----:|:-----|
| `-DBUILD_SHARED_LIBS=ON` | OFF | 构建 DLL |
| `-DNPCORE_ENABLE_CUDA=ON` | OFF | 启用 CUDA GPU |
| `-DBUILD_EXAMPLES=ON` | ON | 编译测试 |

---

## 测试

```bash
# 运行全部测试
cmake --build build -j8
cd build/examples

./test_use_bias       # use_bias 参数
./test_conv1d          # 一维卷积
./test_transformer     # Transformer 编码器
./test_instancenorm    # 实例归一化
./test_serialize       # 模型序列化
./test_trainer         # Trainer API
./test_losses          # 损失函数 + DataLoader
./test_nest_concat     # 序列嵌套 + 分支拼接
./test_onnx            # ONNX 导入导出

# ONNX Python 验证 (需要 pip install onnx onnxruntime)
cd examples/onnx && python verify_onnx.py
```

---

## 文档

| 文档 | 内容 |
|:-----|:-----|
| [快速入门](docs/QuickStart.md) | 安装 + 第一个模型 |
| [API 参考](docs/API.md) | 完整类与方法参考 |
| [层说明](docs/Layers.md) | 30+ 层数学公式与推导 |
| [ONNX 互操作](docs/ONNX.md) | 导入/导出/算子映射/PyTorch 兼容 |
| [数据加载](docs/DataLoader.md) | 图像/序列/CSV/自定义加载 |
| [示例程序](docs/Examples.md) | 22+ 个分类测试 |
| [更新日志](docs/CHANGELOG.md) | 修复 + 新增记录 |
| [路线图](docs/ROADMAP.md) | 后续开发计划 |
| [开发指南](docs/DevGuide.md) | 实现原理 + 编码规范 |
| [自定义模型](docs/CustomModel.md) | 继承 Module 开发新层 |
| [CUDA 指南](docs/CUDA.md) | GPU 编译与内核 |
| [编译指南](docs/BUILD.md) | CMake 选项与源文件列表 |

---

## 项目结构

```
NPCore/
├── Core/                      矩阵引擎 (GEMM, SIMD, CUDA)
├── Layers/
│   ├── Basic/                 Linear Flatten Embedding Dropout Concat
│   ├── Conv/                  Conv1d Conv2d ConvTranspose2d Pool
│   ├── Recurrent/             RNN LSTM GRU
│   ├── Normalization/         BatchNorm LayerNorm GroupNorm InstanceNorm
│   ├── Attention/             MultiHeadAttention PositionalEncoding TransformerEncoder
│   └── Architecture/          Residual ResNetBlock Sequence
├── Activations/               11 种激活函数
├── Losses/                    9 种损失函数
├── Optimizers/                SGD Adam AdamW ... GradientClipping LRScheduler
├── Utils/                     Timer Serializer ONNXModel
├── Cuda/                      CUDA 内核 (可选)
├── examples/                  22+ 个测试
├── assets/                    架构图
├── docs/                      12 篇文档
├── Model.h                    高层 API (FNN CNN Trainer ONNXModel)
├── Autograd.h                 梯度检验
└── NPCore.h                   单头引入
```
