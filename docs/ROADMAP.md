# NPCore 开发路线图

## 一、短期（完善现有模块）

### 1.1 TransformerEncoder / TransformerDecoder
- **理由**: `MultiHeadAttention` 和 `PositionalEncoding` 已经就位，缺少 Encoder/Decoder 的封装
- **工作量**: 中型（~200 行）
- **接口示例**:
```cpp
auto enc = new TransformerEncoder(d_model=512, nhead=8, dim_feedforward=2048, num_layers=6);
// 内部: MHA → Add&Norm → FFN → Add&Norm
```

### 1.2 LayerNorm + Dropout 组合封装
- **理由**: Transformer 中每个 sublayer 都是 `x + Dropout(Sublayer(LayerNorm(x)))`
- **工作量**: 小型（~80 行）
- **可复用 `Residual` 的现有代码**

### 1.3 Conv1d（一维卷积）
- **理由**: 时序信号、音频、NLP 中非常常用
- **工作量**: 小型（~100 行，可复用 Conv2d 的 im2col 逻辑，只是 kernel 变成 1D）
- **接口**:
```cpp
auto conv1d = new Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1);
// 输入: (seq_len, channels) 或 (batch, channels, length)
```

### 1.4 Upsample / PixelShuffle
- **理由**: `ConvTranspose2d` 已有，但 `Upsample(scale_factor)` + `PixelShuffle` 在超分辨率模型中更常用
- **工作量**: 小型（各 ~50 行）

### 1.5 InstanceNorm（实例归一化）
- **理由**: 风格迁移 (Style Transfer) 的核心组件
- **工作量**: 小型（~100 行，可复用 NormLayerBase）
- **接口**:
```cpp
auto inorm = new InstanceNorm2d(channels=64);
```

---

## 二、中期（架构增强）

### 2.1 预定义模型 (Model Zoo)
在 `NPCore/Models/` 下提供常用架构的构建函数：

| 模型 | 用途 | 优先级 |
|------|------|--------|
| `nn::ResNet18/34/50()` | 图像分类 | ★★★ |
| `nn::VGG11/13/16()` | 图像分类 | ★★☆ |
| `nn::UNet()` | 图像分割 | ★★★ |
| `nn::Transformer()` | 序列/文本 | ★★★ |

### 2.2 模型序列化 (Save/Load)
- **当前状态**: 无，训练结果无法保存
- **方案**: 二进制格式 `{num_layers, [shape, data...], ...}` 或 JSON+base64
- **工作量**: 中型（~300 行）
- **接口**:
```cpp
seq.save("model.np");
auto loaded = Sequence::load("model.np");
```

### 2.3 完整的 Autograd 引擎（可选）
- **当前状态**: `numerical_gradient()` + `gradcheck()` 仅用于验证，手写反向传播
- **方案**: 引入计算图 (tape)，对 `MatMul/Add/Mul/Tanh` 等基础 op 定义 `backward`
- **优点**: 新层自动获得梯度，不需要手写 backward
- **缺点**: 需要重构整个库或成为可选的第二套 API
- **建议**: 先做小规模原型评估 ROI

### 2.4 Learning Rate Scheduler 与 Trainer 集成
- **当前状态**: `CosineLR`/`StepLR` 独立存在，但与 `Trainer` 无联动
- **方案**: `Trainer` 增加 `set_lr_scheduler(Scheduler&)`，每个 epoch 自动 step
- **接口**:
```cpp
CosineLR sched(0.01f, 0.0001f, 200, [&](float lr) { optim.set_lr(lr); });
trainer.set_lr_scheduler(sched);
```

---

## 三、长期（工程化）

### 3.1 单元测试框架
- **当前状态**: `examples/unit/gradcheck_test.cpp` 只有梯度检查
- **建议**: 加入 googletest 或 doctest，测试每个 Layer 的 forward shape/backward 梯度/边界条件

### 3.2 Benchmark Suite
- **场景**: Conv2d GEMM vs Winograd、CPU vs GPU、不同 batch size
- **建议**: `examples/benchmark/` 目录，输出 CSV 用于绘图

### 3.3 文档生成 (Doxygen)
- 已有较完整的注释，加 `@param`/`@return` 标签后可自动生成 HTML

### 3.4 ONNX 导出
- **理由**: 方便迁移到推理引擎 (ONNX Runtime / TensorRT)
- **工作量**: 大型
- **建议**: 先锁定常用 op 子集 (Linear/Conv2d/ReLU/MaxPool/BatchNorm)

### 3.5 更多 Loss 函数
| Loss | 用途 |
|------|------|
| `hinge_loss` | SVM |
| `cosine_embedding_loss` | 相似度学习 |
| `triplet_margin_loss` | 度量学习 |
| `ctc_loss` | 语音识别 |

---

## 四、立即可做的小改进

| # | 内容 | 工作量 |
|---|------|--------|
| 1 | `Conv2d` / `Linear` 加 `use_bias=false` 参数 | 5 分钟 |
| 2 | `Softmax` 加 `dim` 参数（当前只支持 row-wise） | 15 分钟 |
| 3 | `Trainer::fit()` callback 加 `epoch` 和 `total_epochs` 参数 | 5 分钟 |
| 4 | 加 `nn::CrossEntropyLoss()` / `nn::BCEWithLogitsLoss()` 高层 wrapper | 10 分钟 |
| 5 | `InMemoryLoader` 加 `normalize()` 方法（自动 Z-score） | 30 分钟 |
| 6 | `ImageFolderLoader` 内置 BMP/PNM 解码器（去掉外部 decoder 依赖） | 1 小时 |
| 7 | `GradientClipping` 与 `Optim::step()` 集成 | 15 分钟 |
| 8 | 所有 Layer 加 `num_params()` 方法（调试时打印参数量） | 10 分钟 |
