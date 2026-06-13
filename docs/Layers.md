# 层说明

## Linear — 全连接层

```
y = W * x + b
```

| 参数 | 说明 |
|------|------|
| `in_features` | 输入维度 |
| `out_features` | 输出维度 |
| `mode` | 权重初始化 (默认 XavierUniform) |
| bias 初始化 | Zeros |

反向传播:
- `dW = grad_output * input^T`
- `db = grad_output`
- `dx = W^T * grad_output`

## Conv2d — 2D 卷积层

```
y[h,w,oc] = Σ_ic Σ_{ki,kj} x[h+ki, w+kj, ic] * W[ki,kj,oc,ic] + b[oc]
```

使用 **im2col + GEMM** 加速:
- Forward: `im2col(input) → W_2d * col → reshape`
- Backward: `dW = G2 * col^T`, `dX = col2im(W^T * G2)`
- 权重缓存: `weight_2d` 懒更新，避免重复变换

## ConvTranspose2d — 转置卷积 (反卷积)

```
Forward:  input_2d → W_T * input_2d → col2im → output + bias
Backward: grad_output → im2col → W * im2col → input_grad
```

| 参数 | 说明 |
|------|------|
| `in_channels` | 输入通道数 |
| `out_channels` | 输出通道数 |
| `kernel_size` | 卷积核大小 (默认 3) |
| `stride` | 步长 (默认 2，实现上采样) |
| `padding` | 填充 (默认 1) |

输出尺寸: `H_out = (H_in - 1) * stride - 2 * padding + kernel_size`

本质是 Conv2d 反向传播的前向版本。广泛用于 U-Net 解码器和 GAN 生成器。
权重形状 `(K, K, C_in * C_out)` 与 Conv2d 保持一致，内部自动转置。

## Sigmoid

```
y = 1 / (1 + exp(-x))
y' = y * (1 - y)
```

- 极端值裁剪: `|x| > 88` 直接返回 0 或 1
- OpenMP 并行 (大矩阵)

## ReLU

```
y = max(0, x)
y' = 1 if x > 0 else 0
```

## SoftMax

```
y_i = exp(x_i - max) / Σ_j exp(x_j - max)
dL/dx_i = y_i * (g_i - Σ_j g_j * y_j)   // O(N) 高效实现
```

- 使用 max-subtraction 防止溢出
- 不再计算完整 N×N Jacobian

## Flatten — 展平层

```
input:  (H, W, C)  →  output: (H*W*C, 1)
```

反向传播恢复原始形状。

## RNN — 循环神经网络

```
h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b_h)
```

| 特性 | 说明 |
|------|------|
| 输入 | `(seq_len, input_size)` 每行一个时间步 |
| 输出 | `(seq_len, hidden_size)` |
| 反向传播 | BPTT (Backpropagation Through Time) |
| 梯度裁剪 | `|grad| > 10 → clip` |
| W_hh 缩放 | 初始化后 ×0.1 (稳定性) |

## LSTM — 长短期记忆

```
f_t = σ(W_f * [x_t; h_{t-1}] + b_f)     // 遗忘门
i_t = σ(W_i * [x_t; h_{t-1}] + b_i)     // 输入门
o_t = σ(W_o * [x_t; h_{t-1}] + b_o)     // 输出门
g_t = tanh(W_g * [x_t; h_{t-1}] + b_g)  // 细胞门
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t        // 细胞状态
h_t = o_t ⊙ tanh(c_t)                   // 隐藏状态
```

| 特性 | 说明 |
|------|------|
| 合并 GEMM | 4 个门合并为 `W(4H, I+H) * [x;h]` |
| 遗忘门 bias | +2 (默认保持记忆) |
| 输入门 bias | -2 (默认不写入) |
| 梯度裁剪 | dh 和 dc 均裁剪到 10 |
| 循环权重 | 初始化后 ×0.1 |

## Sequence — 顺序容器

```cpp
Sequence seq({new Linear(4,8), new Sigmoid(), new Linear(8,4)});
seq.forward(input);
seq.getParams();
seq.cuda();   // 搬到 GPU
seq.eval();   // 推理模式
```

## GRU — 门控循环单元

```
r_t = σ(W_r·[x;h])   重置门
z_t = σ(W_z·[x;h])   更新门
n_t = tanh(W_n·[x; r⊙h])
h_t = (1-z)⊙n + z⊙h
```

2 个门，比 LSTM 参数少 25%。

## MaxPool2d — 最大池化

```
output[i,j,c] = max_{pi,pj} input[i*s+pi, j*s+pj, c]
```

默认 pool=2, stride=2。存储 argmax 用于反向传播。

## AvgPool2d — 平均池化

```
output[i,j,c] = avg_{pi,pj} input[i*s+pi, j*s+pj, c]
```

反向传播将梯度均匀分配给池化窗口内的所有位置。
广泛用于 ResNet 最终分类器前和 DenseNet 过渡层。

## AdaptiveAvgPool2d — 自适应平均池化

```
output size = (output_h, output_w)  // 固定输出尺寸，与输入尺寸无关
```

使用等分区域算法: 每个输出位置对应输入的一个比例区域，对该区域取平均。
关键用途:
- `AdaptiveAvgPool2d(1)` → 全局平均池化 (Global Average Pooling)
- ResNet / ResNeXt 分类器前: 任意尺寸 feature map → 固定尺寸向量

## BatchNorm1d — 批归一化 (1D)

```
y = γ * (x - μ_batch) / √(σ²_batch + ε) + β
```

训练时用 batch 统计，推理时用 running mean/var (动量=0.9)。

## BatchNorm2d — 批归一化 (2D/图像)

```
y_c = γ_c * (x_c - μ_c) / √(σ²_c + ε) + β_c    // c = 通道索引
```

对 `(H, W, C)` 图像张量在空间维度 `H×W` 上归一化，每个通道独立统计。
- Forward: 计算 per-channel μ,σ² → normalize → affine (γ,β)
- Backward: O(H×W×C) 高效计算，使用折叠求和
- train(): 更新 running_mean/running_var (动量 0.9)
- eval(): 使用 running 统计量

## GroupNorm — 分组归一化

```
y = γ_c * (x_c - μ_group(c)) / √(σ²_group(c) + ε) + β_c
```

将 C 个通道分为 G 组，每组内归一化 (H×W×C/G 个元素)。
- 不依赖 batch size: batch=1 也能正常工作
- 适用场景: 目标检测 (Faster R-CNN)、分割 (Mask R-CNN)、GAN (StyleGAN)
- 通常 G=32 或 G=min(32, C)

## LayerNorm — 层归一化

```
y = γ * (x - μ_sample) / √(σ²_sample + ε) + β
```

按样本归一化，不依赖 batch size。适合 RNN/Transformer。

### 归一化对比

| 方法 | 归一化维度 | 依赖 Batch | 典型用途 |
|------|-----------|-----------|---------|
| BatchNorm1d | (N,) per feature | 是 | MLP |
| BatchNorm2d | (H,W) per channel | 是 | CNN, ResNet |
| LayerNorm | (F,) per sample | 否 | RNN, Transformer |
| GroupNorm | (H,W,C/G) per group | 否 | 检测/分割/GAN |

## Dropout — 随机失活

以概率 `rate` 将元素置零，其余放大 `1/(1-rate)` 倍。推理时不做任何操作。

## Embedding — 词嵌入

查表操作：输入整数索引 → 输出对应 embedding 向量。权重矩阵 (vocab_size, embed_dim)。

## MultiHeadAttention — 多头注意力 (Transformer 核心)

```
MultiHead(Q,K,V) = Concat(head_0, ..., head_{h-1}) * W_o
head_i = Attention(Q_i, K_i, V_i) = softmax(Q_i K_i^T / √d_head + mask) * V_i
```

| 参数 | 说明 |
|------|------|
| `d_model` | 模型总维度 (必须被 num_heads 整除) |
| `num_heads` | 注意力头数 (默认 8) |
| `causal` | 因果掩码 (默认 false，用于 decoder 自回归) |

计算流程:
1. Q,K,V = W_q*input, W_k*input, W_v*input (各 d_model×d_model)
2. 拆分为 (d_head, seq_len, num_heads) 多头形状
3. 每头: scores = K^T * Q / √d_head → softmax → V * attn
4. 合并多头 → W_o 投影 → 输出

反向传播: 完整梯度链 (softmax → attention → Q/K/V projection → input)。
8 个参数矩阵 (W_q,W_k,W_v,W_o + 4 bias)。

## GELU — 高斯误差线性单元

```
gelu(x) ≈ 0.5x * (1 + tanh(√(2/π)(x + 0.044715x³)))
```

## Swish / SiLU

```
swish(x) = x * sigmoid(x)
```

## Residual — 残差连接

```cpp
Residual res(new Linear(8,8));  // y = Wx + b + x
```

## ResNetBlock — 残差块

```
Conv3×3 → BN → ReLU → Conv3×3 → BN + skip → ReLU
```

## AdamW — 解耦权重衰减

```cpp
AdamW optim(params, lr=0.001, weight_decay=0.01);
// θ = θ - lr * (m̂/(√v̂+ε) + wd * θ)
```

## CosineLR / StepLR — 学习率调度

```cpp
CosineLR sched(lr_max=0.01, lr_min=1e-4, T_max=500, setter);
StepLR sched(lr=0.01, step_size=50, gamma=0.5, setter);
```

## GradientClipping — 梯度裁剪

```cpp
// 按范数裁剪: 保持梯度方向，限制最大幅度
GradientClipping::clip_by_norm(modules, max_norm=5.0f);
float norm = GradientClipping::clip_by_norm(modules, 5.0f, true); // 返回范数

// 按值裁剪: 逐元素 clamp
GradientClipping::clip_by_value(modules, min_val=-1.0f, max_val=1.0f);
```

RNN/LSTM/Transformer 训练必需品，防止梯度爆炸。
推荐使用 `clip_by_norm` (保持方向)，`clip_by_value` 用于调试极端值。
