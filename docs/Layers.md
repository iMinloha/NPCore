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

## BatchNorm1d — 批归一化

```
y = γ * (x - μ_batch) / √(σ²_batch + ε) + β
```

训练时用 batch 统计，推理时用 running mean/var。

## LayerNorm — 层归一化

```
y = γ * (x - μ_sample) / √(σ²_sample + ε) + β
```

按样本归一化，不依赖 batch size。适合 RNN/Transformer。

## Dropout — 随机失活

以概率 `rate` 将元素置零，其余放大 `1/(1-rate)` 倍。推理时不做任何操作。

## Embedding — 词嵌入

查表操作：输入整数索引 → 输出对应 embedding 向量。权重矩阵 (vocab_size, embed_dim)。

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
