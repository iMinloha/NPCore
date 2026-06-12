# API 参考

## 命名空间

所有 API 在 `CoreNNSpace` 命名空间下。激活函数在 `CoreNNSpace::Activation` 子命名空间。

```cpp
using namespace CoreNNSpace;
```

## Matrix\<T\> — 矩阵类

```cpp
// 构造
Matrix<float> m(rows, cols);              // 2D
Matrix<float> m(rows, cols, channels);    // 3D (用于图像)

// 访问
m.at(i, j)            // 2D 读写
m.at(i, j, ch)        // 3D 读写
m(i, j)               // 2D 只读
m.data_ptr()          // 原始 float* 指针
m.shape()             // Size{row, col, channel}
m.device()            // Device::CPU / Device::GPU

// 运算 (自动 AVX2 SIMD)
m1 * m2               // 矩阵乘法 (packed GEMM 微内核)
m1 + m2               // 逐元素加
m1 - m2               // 逐元素减
m1 & m2               // 逐元素乘 (Hadamard)
m * scalar            // 标量乘
m / scalar            // 标量除
m.Translate()          // 转置
m.sqrt()              // 逐元素开方
m.max()               // 最大值

// 流式赋值
Matrix<float> m(4, 1);
m << 1.0 << 2.0 << 3.0 << 4.0;

// GPU 搬运
m.cuda();             // CPU → GPU
m.cpu();              // GPU → CPU (无 CUDA 时无操作)

// 显示
m.Analysis("title");  // 带格式打印 (±符号, 自适应精度, 科学计数)
```

## 层 API

### 公共接口 (所有层)

```cpp
layer.forward(input);              // 前向传播
layer.backward(grad);              // 反向传播
layer.getParams();                 // 权重指针列表
layer.getAllGrads();               // 梯度指针列表 (与 getParams 对应)
layer.CleanGard();                 // 清理梯度缓存
layer.cuda() / layer.cpu();       // 搬到 GPU / CPU
layer.eval() / layer.train();     // 推理/训练模式
```

### Linear — 全连接

```cpp
Linear(int in_features, int out_features, InitMode = XavierUniform);
// y = W·x + b
```

### Conv2d — 2D 卷积

```cpp
Conv2d(int in_ch, int out_ch, int kernel=3, int stride=1, int padding=0,
       InitMode = XavierUniform);
// 使用 im2col + GEMM 加速
```

### MaxPool2d — 最大池化

```cpp
MaxPool2d(int pool_size=2, int stride=2);
```

### RNN — 循环网络

```cpp
RNN(int input_size, int hidden_size);
// 输入: (seq_len, input_size)  输出: (seq_len, hidden_size)
// h_t = tanh(W_ih·x_t + W_hh·h_{t-1} + b_h)
// 反向传播: BPTT + 梯度裁剪(grad_clip=10)
```

### LSTM — 长短期记忆

```cpp
LSTM(int input_size, int hidden_size);
// 4门合并 GEMM: W(4H, I+H) × [x;h]
// 遗忘门 bias=2, 输入门 bias=-2
// 梯度裁剪: dh 和 dc 均裁剪到 grad_clip=10
```

### GRU — 门控循环

```cpp
GRU(int input_size, int hidden_size);
// 更新门 + 重置门, 参数比 LSTM 少 25%
```

### BatchNorm1d — 批归一化

```cpp
BatchNorm1d(int features);
// 训练用 batch 统计, 推理用 running mean/var
// 输入: (batch, features)
```

### LayerNorm — 层归一化

```cpp
LayerNorm(int features);
// 按样本归一化, 无 running stats
```

### Dropout — 随机失活

```cpp
Dropout(float rate=0.5f);
// 训练时将 rate 比例元素置零, 推理时不做操作
```

### Embedding — 词嵌入

```cpp
Embedding(int vocab_size, int embed_dim);
// 整数索引 → embedding 向量
```

### Residual — 残差连接

```cpp
Residual(Module<float>* sublayer);
// y = sublayer(x) + x
```

### ResNetBlock — 残差块

```cpp
ResNetBlock(int channels, bool use_bn=true);
// Conv3×3→BN→ReLU→Conv3×3→BN + skip → ReLU
```

## 激活函数 (Activation::)

```cpp
Activation::ReLU()         // max(0, x)
Activation::LeakyReLU(α)   // x>0 ? x : αx  (默认α=0.01)
Activation::Sigmoid()      // 1/(1+e^{-x})
Activation::Tanh()         // tanh(x)
Activation::SoftMax()      // e^x / Σe^x
Activation::ELU(α)         // x>0 ? x : α(e^x-1)  (默认α=1)
Activation::SELU()         // 自归一化 ELU
Activation::Softplus()     // ln(1+e^x)
Activation::Mish()         // x·tanh(softplus(x))
Activation::GELU()         // 0.5x(1+tanh(√(2/π)(x+0.0447x³)))
Activation::Swish()        // x·sigmoid(x)
```

## 损失函数

```cpp
mse_loss(pred, target)           // MSE: Σ(p-t)²/n
mse_loss_grad(pred, target)      // MSE 梯度: p-t
l1_loss(pred, target)            // MAE: Σ|p-t|/n
l1_loss_grad(pred, target)       // MAE 梯度: sign(p-t)
smooth_l1_loss(pred, target)     // Huber: |d|<β→d²/2β else |d|-β/2
cross_entropy_loss_grad(lg, t)   // CE + SoftMax: softmax(lg)-t
bce_loss(pred, target)           // Binary CE: -[t·log(p)+(1-t)·log(1-p)]
kl_loss(logp, target)            // KL: Σ t·(log(t)-log(p))
```

## 参数初始化

```cpp
enum InitMode {
    Zeros, Ones, Constant,       // 常数
    Uniform, Gaussian,           // 随机分布
    XavierUniform, HeNormal,     // 深度学习标准
    Identity, Orthogonal         // 特殊矩阵
};

struct InitParams {
    double a=-1, b=1;            // Uniform 范围
    double mu=0, sigma=1;        // Gaussian 参数
    double value=1;              // Constant 值
    float gain=1.0f;             // Orthogonal 增益
};

InitMatrixFunc(matrix, mode, params);
```

## 优化器

```cpp
Optim(params, SGD,       lr)     // 随机梯度下降
Optim(params, Momentum,  lr)     // 动量法 (β=0.9)
Optim(params, Adam,      lr)     // Adam (β1=0.9, β2=0.999)
Optim(params, AdamW,     lr)     // Adam + 解耦权重衰减
Optim(params, RMSProp,   lr)     // RMSProp (β=0.9)
Optim(params, Adagrad,   lr)     // Adagrad
Optim(params, Adadelta,  lr)     // 自适应学习率
Optim(params, NAdam,     lr)     // Nesterov Adam
Optim(params, RAdam,     lr)     // Rectified Adam

optim.step(loss_gradient);
```

## 学习率调度

```cpp
// 余弦退火: lr_max→lr_min, T_max 周期
CosineLR sched(lr_max, lr_min, T_max, setter);
sched.step();                    // 每个 epoch 调用

// 阶梯衰减: 每 step_size 步 × gamma
StepLR sched(lr, step_size, gamma, setter);
```

## Sequence

```cpp
Sequence seq({layer1, layer2, ...});
seq.forward(input);
seq.getParams();   // 收集所有子层参数
seq.cuda();        // 搬到 GPU
seq.eval();        // 推理模式
```

## 梯度检验

```cpp
#include "Autograd.h"

// 数值梯度 (中心差分)
Matrix<float> ng = numerical_gradient<float>(loss_fn, x, 1e-4f);

// 对比分析梯度 vs 数值梯度 (阈值 1e-2)
bool ok = gradcheck(analytical, numerical, 1e-2f, "label");
```
