# NPCore 开发路线图

## 已完成 (v1.1)

- [x] `Conv1d` 一维卷积
- [x] `TransformerEncoder` + `PositionalEncoding`
- [x] `InstanceNorm1d/2d`
- [x] `Sequence` 继承 `Module<float>` 支持嵌套
- [x] `Concat` 分支拼接层
- [x] `use_bias` 参数 (Linear/Conv2d/ConvTranspose2d/Conv1d)
- [x] ONNX 完整导入/导出 (`ONNXModel` OOP API)
- [x] 模型序列化 (`save_model`/`load_model_weights`)
- [x] `modules()` 递归展平叶子层
- [x] Loss 函数补全 (`cross_entropy_loss`, `bce_loss_grad`, `smooth_l1_loss_grad`)
- [x] 优化器 factory 扩展到 8 个
- [x] 22+ 个分类测试

## 短期

- [ ] `Conv2d`/`Linear` 序列化时保留 `kernel_size`/`stride`/`padding` 元数据
- [ ] `TransformerDecoder` + cross-attention
- [ ] `Upsample` / `PixelShuffle` 层
- [ ] ONNX shape 推断 (当前使用动态 dim)
- [ ] ONNX `onnxruntime` 端到端推理验证
- [ ] `Embedding` ↔ ONNX `Gather` 导入导出
- [ ] `Trainer` 集成 `CosineLR`/`StepLR`
- [ ] `FNN`/`CNN` builder 加 dropout 参数

## 中期

- [ ] 预定义模型: `ResNet18/34/50`, `VGG`, `UNet`
- [ ] 模型文件格式统一 (`.np` header + version)
- [ ] googletest 单元测试框架
- [ ] `MatMul` + `Add` → 残差连接 ONNX 还原
- [ ] `ConvTranspose2d` ↔ ONNX `ConvTranspose`
- [ ] 完整 Doxygen 文档生成

## 长期

- [ ] 完整 Autograd 引擎 (计算图)
- [ ] ONNX Runtime 集成推理
- [ ] 分布式训练 (MPI)
- [ ] 移动端部署 (ARM NEON 优化)
- [ ] RNN/LSTM/GRU ONNX 导入导出
