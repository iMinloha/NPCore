# NPCore ONNX 互操作

## API

```cpp
#include "NPCore.h"
using namespace NPCore::nn;
```

| 方法 | 方向 | 说明 |
|------|------|------|
| `ONNXModel::load(path)` | 导入 | 解析 ONNX 文件 → ONNXModel |
| `onnx.to_sequence()` | 导入 | ONNXModel → Sequence（可推理/训练） |
| `ONNXModel::from_sequence(seq, shape, name)` | 导出 | Sequence → ONNXModel |
| `onnx.save(path)` | 导出 | 写入 ONNX 文件 |
| `onnx.nodes()` / `weights()` / `inputs()` / `outputs()` | 查看 | 元数据 |

## 使用示例

### 导出：NPCore → ONNX

```cpp
Sequence model({
    new Linear(2, 8), new Activation::ReLU(),
    new Linear(8, 2), new Activation::SoftMax()
});
auto onnx = ONNXModel::from_sequence(model, {1, 2}, "MLP");
onnx.save("model.onnx");
```

### 导入：ONNX → NPCore

```cpp
auto onnx = ONNXModel::load("model.onnx");
// 查看结构
for (auto& n : onnx.nodes())
    printf("%s: %zu in → %zu out\n", n.op_type.c_str(),
           n.inputs.size(), n.outputs.size());
// 转为可运行模型
Sequence model = onnx.to_sequence();
model.eval();
auto out = model.forward(input);
```

### 验证 ONNX 文件

```bash
python examples/onnx/verify_onnx.py
```

## 算子支持

| ONNX Op | NPCore 层 | 导出 | 导入 |
|---------|-----------|:----:|:----:|
| `Gemm` | `Linear` | ✅ | ✅ |
| `Conv` (2D) | `Conv2d` | ✅ | ✅ |
| `Conv` (1D) | `Conv1d` | ✅ | ✅ |
| `Relu` | `Activation::ReLU` | ✅ | ✅ |
| `Sigmoid` | `Activation::Sigmoid` | ✅ | ✅ |
| `Tanh` | `Activation::Tanh` | ✅ | ✅ |
| `Softmax` | `Activation::SoftMax` | ✅ | ✅ |
| `Gelu` | `Activation::GELU` | ✅ | ✅ |
| `LeakyRelu` | `Activation::LeakyReLU` | ✅ | ✅ |
| `MaxPool` | `MaxPool2d` | ✅ | ✅ |
| `AveragePool` | `AvgPool2d` | ✅ | ✅ |
| `Flatten` | `Flatten` | ✅ | ✅ |
| `Reshape` | `Flatten` | — | ✅ |
| `BatchNormalization` | `BatchNorm1d/2d` | ✅ | ✅ |
| `Concat` | `Concat` | ✅ | ✅ |
| 嵌套展平 | `Sequence` 递归 | ✅ | — |

## 模型结构支持

- **线性链** ✅
- **嵌套 Sequence** ✅ — 递归展平为 ONNX 节点
- **分支 (Concat)** ✅ — 导出为 ONNX `Concat` 节点
- **残差 (Residual)** — 手动包装后导出

## 与 PyTorch 互通

### PyTorch → ONNX → NPCore

```python
import torch
model = torch.nn.Linear(4, 8)
model.eval()
dummy = torch.randn(1, 4)
torch.onnx.export(model, dummy, "linear.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=11)
```

```cpp
auto onnx = ONNXModel::load("linear.onnx");
Sequence model = onnx.to_sequence();
```

### 已知差异

| 特性 | PyTorch | NPCore | 影响 |
|------|---------|--------|------|
| 张量布局 | `(N,C,H,W)` 通道在前 | `(H,W,C)` 通道在后 | 权重还原正确，输入需 permute |
| 线性层 | `(batch, feat)` 行向量 | `(feat, 1)` 列向量 | 权重转置一致 |
| BatchNorm | 导出时折叠进 Conv | 保留独立层 | 导入时作为 BatchNorm1d |
| GELU/Attention | opset<20 时分解为原语 | — | 无法从 ONNX 还原为高层层 |
| Dropout(eval) | 导出为 `Identity` | — | 无操作层 |
| opset | 推荐 11 | 兼容 8/11/13/14 | 不影响权重 |

## 架构示例

```
examples/onnx/
├── test_onnx.cpp      # C++ 测试 (导出/导入/嵌套/分支)
└── verify_onnx.py     # Python ONNX 验证脚本
```
