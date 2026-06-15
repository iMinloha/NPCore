# NPCore ONNX 支持文档

## 概述

NPCore 通过 `nn::ONNXModel` 类提供完整的 ONNX 导入/导出功能，支持与 PyTorch 等主流框架的模型互操作。

```cpp
#include "NPCore.h"
using namespace NPCore::nn;
```

---

## 1. ONNXModel 类 API

### 1.1 导出：Sequence → ONNX 文件

```cpp
// 构建模型
Sequence model({
    new Conv2d(1, 8, 3, 1, 0),
    new Activation::ReLU(),
    new MaxPool2d(2, 2),
    new NPCore::Flatten(),
    new Linear(13*13*8, 10)
});

// 导出 ONNX
auto onnx = ONNXModel::from_sequence(model, {1, 1, 28, 28}, "MyCNN");
onnx.save("model.onnx");
```

### 1.2 导入：ONNX 文件 → Sequence

```cpp
// 从文件加载
auto onnx = ONNXModel::load("model.onnx");

// 查看模型信息
onnx.graph_name();        // "MyCNN"
onnx.producer_name();     // "NPCore"
onnx.num_nodes();         // 节点数
onnx.weights().size();    // 权重张量数

// 转换为可运行的 Sequence
Sequence model = onnx.to_sequence();

// 前向传播
model.eval();
Matrix<float> input(28, 28, 1);
Matrix<float> output = model.forward(input);
```

### 1.3 元数据查看

```cpp
auto onnx = ONNXModel::load("model.onnx");

// 图结构
for (auto& node : onnx.nodes())
    printf("%s: %zu inputs -> %zu outputs\n",
           node.op_type.c_str(),
           node.inputs.size(), node.outputs.size());

// 输入/输出名
for (auto& name : onnx.inputs())  printf("in:  %s\n", name.c_str());
for (auto& name : onnx.outputs()) printf("out: %s\n", name.c_str());

// 权重张量
for (auto& w : onnx.weights())
    printf("weight '%s': [", w.name.c_str());
    for (auto d : w.dims) printf("%lld ", (long long)d);
    printf("] %zu floats\n", w.data.size());
```

---

## 2. 支持的 ONNX 算子

### 2.1 导出支持 (NPCore → ONNX)

| NPCore 层 | ONNX Op | 备注 |
|-----------|---------|------|
| `Linear` | `Gemm` | transB=1, alpha=1, beta=1 |
| `Conv2d` | `Conv` | kernel_shape 属性 |
| `Conv1d` | `Conv` | kernel_shape 属性 (1D) |
| `ReLU` | `Relu` | |
| `Sigmoid` | `Sigmoid` | |
| `Tanh` | `Tanh` | |
| `SoftMax` | `Softmax` | axis=1 |
| `GELU` | `Gelu` | ONNX opset ≥ 20 |
| `LeakyReLU` | `LeakyRelu` | alpha 属性 |
| `MaxPool2d` | `MaxPool` | kernel_shape + strides |
| `AvgPool2d` | `AveragePool` | kernel_shape + strides |
| `Flatten` | `Flatten` | axis=1 |
| `BatchNorm1d` | `BatchNormalization` | epsilon, running mean/var=0/1 |
| `BatchNorm2d` | `BatchNormalization` | 同上 |

**未支持的层**（导出时跳过）：
- `Dropout` — 推理时恒等，无需导出
- `Embedding` — ONNX 对应 `Gather`，暂未实现
- `RNN/LSTM/GRU` — ONNX 对应 `RNN/LSTM/GRU`，结构和权重布局不同
- `LayerNorm/GroupNorm/InstanceNorm` — 可扩展
- `MultiHeadAttention/TransformerEncoder` — 可扩展
- `Residual/ResNetBlock` — 结构复杂，建议拆分后导出

### 2.2 导入支持 (ONNX → NPCore)

| ONNX Op | NPCore 层 | 备注 |
|---------|-----------|------|
| `Gemm` | `Linear` | 自动识别 transB 并转置权重 |
| `Conv` | `Conv2d` / `Conv1d` | 根据 weight dims 自动判断 1D/2D |
| `Relu` | `Activation::ReLU` | |
| `Sigmoid` | `Activation::Sigmoid` | |
| `Tanh` | `Activation::Tanh` | |
| `Softmax` | `Activation::SoftMax` | |
| `Gelu` | `Activation::GELU` | |
| `LeakyRelu` | `Activation::LeakyReLU` | alpha 属性可配置 |
| `MaxPool` | `MaxPool2d` | 默认 pool=2, stride=2 |
| `AveragePool` | `AvgPool2d` | 默认 pool=2, stride=2 |
| `Flatten` | `Flatten` | |
| `Reshape` | `Flatten` | PyTorch 的 view/reshape 兼容 |
| `BatchNormalization` | `BatchNorm1d` | gamma/beta 权重恢复 |

**不支持的算子**（导入时跳过）：
- `Shape`, `Gather`, `Unsqueeze`, `Concat`, `Constant` — PyTorch 辅助节点，不影响推理
- `Add` (残差连接) — 可扩展
- `MatMul` — 可扩展
- `ReduceMean`, `Mul`, `Div` — 可扩展

---

## 3. PyTorch 兼容性

### 3.1 已验证的 PyTorch 模型结构

```
SmallCNN (已验证通过):
  Conv2d(1→8, k=3, p=1)  →  BN  →  ReLU  →  MaxPool(2)
  Conv2d(8→16, k=3, p=1) →  BN  →  ReLU  →  MaxPool(2)
  Flatten  →  Linear(784→128)  →  ReLU  →  Linear(128→10)  →  Softmax
```

导出命令：
```bash
python examples/onnx/export_torch_model.py
```

导入验证：
```bash
build/examples/test_onnx_import.exe
# 5 PASS, 0 FAIL
```

### 3.2 可支持的 PyTorch 结构

| PyTorch 结构 | ONNX 表示 | NPCore 还原 |
|-------------|----------|-------------|
| `nn.Linear(in, out)` | `Gemm` | `Linear(in, out)` |
| `nn.Conv2d(in, out, k)` | `Conv` | `Conv2d(in, out, k)` |
| `nn.Conv1d(in, out, k)` | `Conv` (1D) | `Conv1d(in, out, k)` |
| `nn.ReLU()` | `Relu` | `Activation::ReLU` |
| `nn.Sigmoid()` | `Sigmoid` | `Activation::Sigmoid` |
| `nn.Tanh()` | `Tanh` | `Activation::Tanh` |
| `nn.Softmax(dim)` | `Softmax` | `Activation::SoftMax` |
| `nn.GELU()` | `Gelu` | `Activation::GELU` |
| `nn.LeakyReLU()` | `LeakyRelu` | `Activation::LeakyReLU` |
| `nn.MaxPool2d(2)` | `MaxPool` | `MaxPool2d(2,2)` |
| `nn.AvgPool2d(2)` | `AveragePool` | `AvgPool2d(2,2)` |
| `nn.Flatten()` / `view` | `Flatten` / `Reshape` | `Flatten` |
| `nn.BatchNorm1d` | `BatchNormalization` | `BatchNorm1d` |
| `nn.BatchNorm2d` | `BatchNormalization` | `BatchNorm1d` |

### 3.3 PyTorch 导出建议

```python
import torch

# 1. 模型定义（避免使用不支持的 op）
model.eval()

# 2. 导出 ONNX
dummy = torch.randn(1, 1, 28, 28)  # batch=1
torch.onnx.export(
    model, dummy, "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,  # 推荐 11
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

# 3. 验证
import onnx
onnx.checker.check_model(onnx.load("model.onnx"))
```

### 3.4 已知差异

| 特性 | PyTorch | NPCore | 影响 |
|------|---------|--------|------|
| 张量布局 | `(N,C,H,W)` 通道在前 | `(H,W,C)` 通道在后 | 权重还原正确，输入需手动 permute |
| 线性层 | `(batch, features)` 行向量 | `(features, 1)` 列向量 | 权重转置，Flatten 后形状可能不同 |
| BatchNorm | 导出时折叠进 Conv | 保留为独立层 | 导入时 BN 作为独立 BatchNorm1d 还原 |
| opset | 默认最新 | 推荐 opset 11 | 旧版 opset 兼容性更好 |

---

## 4. 权重序列化对比

NPCore 提供两套持久化方案：

| 特性 | `save_model` / `load_model_weights` | `ONNXModel` |
|------|-------------------------------------|-------------|
| 格式 | 二进制 (NPCM magic) | ONNX protobuf |
| 跨框架 | 仅 NPCore | PyTorch / ONNX Runtime / Netron |
| 精度 | 完全一致 (raw float) | 完全一致 (raw float) |
| 速度 | 极快 | 需要 protobuf 编解码 |
| 用途 | 训练 checkpoint | 部署 / 交换 |

---

## 5. 完整示例

### 5.1 NPCore 训练 → ONNX 部署

```cpp
// 训练模型
auto net = nn::CNN({1, 8, 16}, 3, nn::ReLU, 10);
nn::Trainer trainer(net, nn::CrossEntropy, nn::Adam(0.001f));
trainer.fit(loader, 50);

// 导出 ONNX 并在 Python 中加载
auto onnx = nn::ONNXModel::from_sequence(net, {1, 1, 28, 28}, "TrainedCNN");
onnx.save("deploy.onnx");
```

```python
# 在 Python 中推理
import onnxruntime as ort
sess = ort.InferenceSession("deploy.onnx")
out = sess.run(None, {"input": image.numpy()})
```

### 5.2 PyTorch 训练 → NPCore 加载

```python
# PyTorch 导出
torch.onnx.export(trained_model, dummy_input, "from_torch.onnx", opset_version=11)
```

```cpp
// NPCore 加载
auto onnx = nn::ONNXModel::load("from_torch.onnx");
Sequence model = onnx.to_sequence();
model.eval();
auto result = model.forward(input);
```

### 5.3 ONNX 文件可视化

```bash
# 使用 Netron 查看模型结构
netron model.onnx

# Python 验证合法性
python -c "import onnx; onnx.checker.check_model(onnx.load('model.onnx')); print('OK')"
```

---

## 6. 限制与路线图

### 当前限制
- 不支持 RNN/LSTM/GRU 的 ONNX 导入导出
- 不支持 Attention/Transformer 结构的自动转换
- Reshape 导入时统一映射为 Flatten（丢失具体 shape 信息）
- 输入/输出使用动态维度 (dim_param)，部分推理引擎需要固定 shape

### 计划支持
- `Embedding` ↔ `Gather`
- `MatMul` + `Add` → 残差连接的还原
- `ConvTranspose2d` ↔ `ConvTranspose`
- 完整的 TypeProto shape 推断
