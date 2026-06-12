# API 参考

## 高层 API (`Model.h`)

### 网络工厂

```cpp
// FNN: 全连接网络
auto fnn = nn::FNN({input, h1, h2, ..., output}, activation);

// CNN: 卷积网络 (3×3 卷积核，自动 Flatten + Linear)
auto cnn = nn::CNN({in_channels, c1, c2, ...}, kernel_size, activation, output_classes);
```

### 激活函数

| 简写 | 实际类 |
|------|--------|
| `nn::Sigmoid()` | `Activation::Sigmoid` |
| `nn::ReLU()` | `Activation::ReLU` |
| `nn::SoftMax()` | `Activation::SoftMax` |
| `nn::Flatten()` | `Flatten` |

### 优化器

```cpp
Optim(params, SGD,     lr)     // 随机梯度下降
Optim(params, Momentum, lr)    // 动量法
Optim(params, Adam,     lr)    // Adam (推荐)
Optim(params, RMSProp,  lr)    // RMSProp
Optim(params, Adagrad,  lr)    // Adagrad

// 简写
nn::Adam(0.001)
nn::SGD(0.01)
nn::RMSProp(0.01)
```

### Trainer

```cpp
nn::Trainer trainer(model, loss, optimizer);

// 单样本训练
trainer.fit(input, target, epochs, callback);

// 多样本交替训练
trainer.fit(samples, epochs, callback);

// callback: void(int epoch, float loss)
```

### 损失函数

```cpp
nn::MSE          // 均方误差
nn::CrossEntropy // 交叉熵 (与 SoftMax 配合)
```

## 底层 API (`CorePP.h`)

### Matrix\<T\>

```cpp
Matrix<float> m(rows, cols);              // 2D
Matrix<float> m(rows, cols, channels);    // 3D

// 访问
m.at(i, j)         // 2D 元素
m.at(i, j, ch)     // 3D 元素
m(i, j)            // 只读 2D
m.data_ptr()       // 原始指针

// 运算
m1 * m2            // 矩阵乘法 (GEMM)
m1 + m2            // 逐元素加
m1 - m2            // 逐元素减
m1 & m2            // 逐元素乘
m.Translate()       // 转置
m.Analysis("name") // 打印矩阵

// 流式赋值
Matrix<float> m(4, 1);
m << 1.0 << 2.0 << 3.0 << 4.0;
```

### Layer 层

```cpp
// Linear: 全连接
Linear in_out(input_size, output_size, init_mode);
// init_mode: XavierUniform (默认), HeNormal, Uniform, Gaussian, Zeros

// Conv2d: 2D 卷积
Conv2d conv(in_ch, out_ch, kernel, stride, padding, init_mode);

// Activation
Activation::ReLU()
Activation::Sigmoid()
Activation::SoftMax()

// Flatten: 展平 (Conv → Linear 之间)
Flatten()

// RNN: 循环网络
RNN rnn(input_size, hidden_size);

// LSTM: 长短期记忆
LSTM lstm(input_size, hidden_size);

// Sequence: 顺序容器
Sequence seq({layer1, layer2, ...});
seq.forward(input);
seq.getParams();  // 返回所有 Module* 指针
```

### 权重初始化

```cpp
enum InitMode {
    Zeros, Ones, Constant, Uniform, Gaussian,
    XavierUniform, HeNormal, Identity, Orthogonal
};

struct InitParams {
    double a = -1, b = 1;      // Uniform 范围
    double mu = 0, sigma = 1;   // Gaussian 参数
    double value = 1;           // Constant 值
    float  gain = 1.0f;         // Orthogonal 增益
};

InitMatrixFunc(matrix, mode, params);
```

### 梯度检验

```cpp
#include "Autograd.h"

// 数值梯度
Matrix<float> ng = numerical_gradient<float>(loss_fn, x, epsilon);

// 梯度对比
bool ok = gradcheck(analytical, numerical, threshold, "label");
```
