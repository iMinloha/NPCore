# 示例程序指南

14 个分类示例，每个都打印输入/输出与期望值的对比，用于验证实现正确性与精度分析。

## 目录结构

```
examples/
├── unit/                 test_gradcheck      解析梯度 vs 数值梯度
├── basic/                test_fnn            全连接网络训练
├── conv/                 test_cnn            卷积分类
│                         test_convtranspose2d 转置卷积上采样
│                         test_pooling         平均/自适应池化
├── recurrent/            test_rnn / lstm / gru
├── normalization/        test_batchnorm2d    2D 批归一化
│                         test_groupnorm      分组归一化
├── attention/            test_mha            多头注意力
├── optimizers/           test_gradclip       梯度裁剪
├── architecture/         test_resnet         残差网络
└── standalone/           demo_app            完整 API 演示 (外部复用)
                          demo_classifier     CNN 图像分类器
```

运行: `cmake --build _build && ./_build/examples/test_mha`

### 独立项目 (standalone/)
演示如何将 NPCore 作为外部库使用:

```bash
# 方式 A: 从源码自动编译
cd examples/standalone && mkdir _b && cd _b
cmake .. -DNPCORE_SOURCE_DIR=../../
cmake --build .

# 方式 B: 使用已安装的库
cmake .. -DNPCore_DIR=/path/to/install/lib/cmake/NPCore
cmake --build .
```

| Demo | 内容 |
|------|------|
| demo_app | FNN+梯度裁剪、CNN+BatchNorm+Pooling、Transformer、DataLoader |
| demo_classifier | 8层 CNN: Conv->BN->ReLU->Pool->Conv->BN->ReLU->AdaptivePool->Flatten->Linear, 对角/反对角分类 |



---

## unit/test_gradcheck

验证 Linear 层解析梯度与数值梯度的误差。

**测试方法:** 中心差分法计算数值梯度，与 backward() 返回的解析梯度逐元素对比。

**期望输出:**
```
[gradcheck] PASS-Linear dL/dx (max abs err: <0.001, max rel err: <0.001)
Gradient check PASSED!
```

---

## basic/test_fnn

4->8->16->8->4 全连接网络，Adam 优化器，300 轮训练。

**期望输出:** 网络将输入 `[3,4,2,1]` 拟合到目标 `[1,0,0,0]`。最终 MSE < 0.001。

---

## conv/test_cnn

Conv->ReLU->Pool->Flatten->Linear 流水线。

**期望输出:** CNN Test COMPLETED。验证端到端特征提取和分类流程。

---

## conv/test_convtranspose2d

输入 4x4x4 -> 输出 7x7x2 (stride=2 上采样)。

**期望输出:**
```
Expected output shape: (7, 7, 2)
Actual output shape:   (7, 7, 2)
[PASS] Output spatial size = (H_in-1)*S - 2P + K
[PASS] Output channels = C_out
[PASS] Forward output is finite
[PASS] Forward output is non-zero
[PASS] Input gradient shape matches input (4,4,4)
[PASS] Weight gradient computed
[PASS] Bias gradient computed
```
全部 10 项检查通过。公式: H_out = (4-1)*2 - 2*1 + 3 = 7。

---

## conv/test_pooling

### AvgPool2d
4x4x2 -> 2x2x2 (pool=2, stride=2)。

**期望输出:** 每个输出单元 = 对应 2x2 输入区域的平均值。与手动计算逐元素对比，最大误差 < 1e-4。
梯度均匀分配: `dL/dx[i,j] = 1/4 * grad_output`（每个输入位置获得 1/4 的输出梯度）。

### AdaptiveAvgPool2d
5x3x3 -> 1x1x3（全局平均池化）。

**期望输出:**
- ch0 (全 1.0): 输出 = 1.0
- ch1 (1..15): 输出 = 8.0
- ch2 (全 7.0): 输出 = 7.0
- 反向传播验证: `dL/dx[0,0,0] = 1.0/15`, `dL/dx[0,0,1] = 2.0/15`

---

## recurrent/test_rnn / test_lstm / test_gru

序列建模，验证 BPTT 反向传播正确性。

**期望输出:** 各测试输出 "COMPLETED"，无断言失败。RNN/LSTM/GRU 内部已包含手动梯度验证。

---

## normalization/test_batchnorm2d

4x4x4 输入，4 个通道: 常量/渐变/常量/渐变。

**期望输出:**
| 检查 | 期望 | 实际 |
|------|------|------|
| Ch0 (全 5.0) | mu≈0, sigma^2≈0 | mu≈0, sigma^2≈0 |
| Ch1 (0..15) | mu≈0, sigma^2≈1 | mu≈0, sigma^2≈1.000 |
| Ch2 (全 -3.0) | mu≈0, sigma^2≈0 | mu≈0, sigma^2≈0 |
| Ch3 (30..45) | mu≈0, sigma^2≈1 | mu≈0, sigma^2≈1.000 |
| 输出形状 | (4,4,4) | (4,4,4) |
| dgamma/dbeta | 非空 | 已计算 |

---

## normalization/test_groupnorm

2 组 x 3 通道，组 0 值域 0..17，组 1 值域 100..117。

**期望输出:**
| 检查 | 期望 | 实际 |
|------|------|------|
| Group 0 | mu≈0, sigma^2≈1 | mu≈0, sigma^2≈1 |
| Group 1 | mu≈0, sigma^2≈1 | mu≈0, sigma^2≈1 |
| 输出形状 | (2,3,6) | (2,3,6) |
| dgamma/dbeta | 非空 | 已计算 |
| 跨组独立性 | 两组独立归一化 | 验证通过 |

---

## attention/test_mha

seq=4, d_model=32, heads=4, causal=true。

**期望输出:**
| 检查 | 期望 |
|------|------|
| 输出形状 | (4, 32) = 输入形状 |
| 前向有限性 | sum 为有限非零值 |
| 反向梯度形状 | (4, 32) |
| 8 个参数梯度 | 全部非空且非零 |

全部 9 项检查通过。

---

## optimizers/test_gradclip

**clip_by_value:** 生成的梯度值超出 [-0.5, 0.5] -> 裁剪后全部在范围内。

**clip_by_norm:** 梯度原始 L2 范数 > 3.0 -> 裁剪后范数 = 3.0（保持方向，限制幅度）。

---

## architecture/test_resnet

Conv->ResBlock->Pool->Flatten->Linear 架构。

**期望输出:** "ResNet Test COMPLETED"。

---

## data/test_dataloader

自定义数据集 30 样本，7:3 训练/测试划分，RMSProp 优化。

**期望输出:** Test accuracy > 85%。
