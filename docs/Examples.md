# 示例程序指南

CorePP 提供 14 个分类示例程序，覆盖从基础层到 Transformer 的所有组件。
每个示例都打印输入/输出/期望值的精度分析，可直接运行验证实现正确性。

## 目录结构

```
examples/
├── unit/                 单元测试
│   └── test_gradcheck    数值梯度 vs 解析梯度对比
├── basic/                基础层
│   └── test_fnn          全连接网络训练
├── conv/                 卷积与池化
│   ├── test_cnn          卷积 + 池化 + 全连接
│   ├── test_convtranspose2d  转置卷积 (U-Net 解码器)
│   └── test_pooling      AvgPool2d + AdaptiveAvgPool2d
├── recurrent/            循环网络
│   ├── test_rnn          Elman RNN
│   ├── test_lstm         长短期记忆
│   └── test_gru          门控循环单元
├── normalization/        归一化层
│   ├── test_batchnorm2d  2D 批归一化 (CNN)
│   └── test_groupnorm    分组归一化 (检测/分割)
├── attention/            注意力机制
│   └── test_mha          多头注意力 (Transformer)
├── optimizers/           优化器工具
│   └── test_gradclip     梯度裁剪
├── architecture/         架构组件
│   └── test_resnet       残差网络
└── data/                 数据加载
    └── test_dataloader   自定义数据集 + 训练/测试划分
```

## 运行

```bash
cmake --build _build
./_build/examples/test_mha      # 运行单个测试
# 批量运行: for exe in _build/examples/test_*.exe; do $exe; done
```

## 各示例说明

### test_gradcheck — 梯度正确性验证
验证 Linear 层解析梯度与数值梯度的误差 < 1e-3。
使用中心差分法计算数值梯度，逐元素对比。

### test_fnn — 全连接网络
4->8->16->8->4 网络，Adam 优化器，300 轮训练。
验证网络能将输入 `[3,4,2,1]` 拟合到目标 `[1,0,0,0]`。

### test_cnn — 卷积网络
Conv->ReLU->Pool->Flatten->Linear 流水线。
验证卷积特征提取 + 分类的端到端流程。

### test_convtranspose2d — 转置卷积
输入 4×4×16 -> 输出 7×7×8 (上采样)。
验证公式 `H_out = (H_in-1)×S - 2P + K` 的精度。
打印输入/输出/权重的详细分析。

### test_pooling — 池化层
- **AvgPool2d**: 4×4×2 -> 2×2×2，与手动计算逐元素对比，验证梯度均匀分配
- **AdaptiveAvgPool2d**: 5×3×3 -> 1×1×3 (全局池化)，验证不同尺寸输入的自适应

### test_rnn / test_lstm / test_gru — 循环网络
序列建模任务，验证 BPTT 反向传播和门控机制的正确性。

### test_batchnorm2d — 批归一化
验证 per-channel 统计量 (μ,σ²)，训练模式 vs 推理模式对比。
包含常量通道 (σ²=0) 和变化通道 (σ²>0) 的边界测试。

### test_groupnorm — 分组归一化
2 组 × 3 通道，验证组间独立归一化。
组 0 (值域 0..17) 和组 1 (值域 100..117) 各自归一化到 μ≈0,σ²≈1。

### test_mha — 多头注意力
4 头 × 8 维 = 32 维模型，因果掩码。
验证 Q/K/V 投影、缩放点积注意力、多头合并、输出投影的完整梯度链。

### test_gradclip — 梯度裁剪
- clip_by_value: 验证所有梯度值 clamp 到 [min,max]
- clip_by_norm: 验证 L2 范数缩放因子精确性

### test_resnet — 残差网络
Conv->ResBlock->Pool->Flatten->Linear 架构，验证跳跃连接。

### test_dataloader — 数据加载
自定义数据集训练，演示 Train/Test 划分和批量迭代。
