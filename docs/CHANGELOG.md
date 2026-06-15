# Changelog

## v1.1 — 2026-06-15

### 新增功能

| 功能 | 文件 | 说明 |
|------|------|------|
| `Sequence` 嵌套 | `Sequence.h/.cpp` | 继承 `Module<float>`，可放入另一个 Sequence |
| `Concat` 分支层 | `Basic/Concat.h/.cpp` | 多分支拼接，支持 Inception/U-Net |
| `Conv1d` | `Conv/Conv1d.h/.cpp` | 一维卷积 (时序信号) |
| `TransformerEncoder` | `Attention/TransformerEncoder.h/.cpp` | 多层 Transformer 编码器 |
| `PositionalEncoding` | `Attention/PositionalEncoding.h/.cpp` | 正弦余弦位置编码 |
| `InstanceNorm1d/2d` | `Normalization/InstanceNorm.h/.cpp` | 实例归一化 (风格迁移) |
| `ONNXModel` | `Utils/ONNXModel.h/.cpp` | ONNX 导入/导出 OOP API |
| `Serializer` | `Utils/Serializer.h/.cpp` | 模型序列化 `save_model`/`load_model_weights` |
| `use_bias` 参数 | `Linear` `Conv2d` `ConvTranspose2d` `Conv1d` | 可选偏置 |
| `cross_entropy_loss` | `Losses/Loss.h/.cpp` | 完整 CE loss 值函数 |
| `bce_loss_grad` | `Losses/Loss.h/.cpp` | BCE 梯度函数 |
| `smooth_l1_loss_grad` | `Losses/Loss.h/.cpp` | Huber 梯度函数 |
| 优化器工厂 | `Model.h/.cpp` | 8 个 `nn::SGD/Momentum/Adam/...` 工厂函数 |
| `modules()` | `Module.h/.cpp` | 递归获取叶子层 (替代 `getParams` 传给 Optimizer) |

### Bug 修复

| 问题 | 文件 | 修复 |
|------|------|------|
| `Tanh()` 返回 Sigmoid | `Model.cpp` | `Activation::Sigmoid()` → `Tanh()` |
| Trainer 参数不更新 | `Model.h/cpp` `Optimizer.h/cpp` | `modules()` + `set_params()` |
| `eval/train` 非 virtual | `Module.h` `Residual.h` `ResNetBlock.h` | 基类加 `virtual`，子类加 `override` |
| GRU 重置门未实现 | `GRU.cpp` | 正向前向+正确反向传播 |
| `Matrix::operator+(float)` 忽略 channel | `Matrix.cpp` | `row*col` → `row*col*channel` |
| `Matrix::operator/(float)` 忽略 channel | `Matrix.cpp` | 同上 |
| `SequenceLoader` 参数废弃 | `DataLoader.h/.cpp` | 维度校验 |
| `loss_val` 无视 LossType | `Model.cpp` | 正确分发 MSE/CrossEntropy |
| ONNX Gemm `transB` 维度 | `ONNXModel.cpp` | 移除错误的 swap |
| Concat 双重释放 | `ONNXModel.cpp` | `to_sequence()` 从 layers 移除分支 |

---

## v1.0 — 初始版本

- 核心矩阵引擎 (GEMM + AVX2 SIMD)
- 基础层: Linear, Conv2d, ConvTranspose2d, Pool, Flatten
- 循环层: RNN, LSTM, GRU
- 归一化: BatchNorm1d/2d, LayerNorm, GroupNorm
- 注意力: MultiHeadAttention
- 架构: Residual, ResNetBlock, Sequence
- 11 种激活函数, 6 种损失函数
- 8 种优化器 + LR Scheduler
- 数值梯度检验
- CUDA 后端 (可选)
- 完整文档
