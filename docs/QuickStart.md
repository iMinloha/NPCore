# 快速入门

## 安装依赖

- **编译器**: MinGW-w64 (GCC 12+) 或 MSVC 2022
- **CMake**: 3.18+
- **可选**: CUDA Toolkit 11.0+, Visual Studio Build Tools

## 构建

```bash
# 克隆并构建
cd CorePP
cmake -G "MinGW Makefiles" -B _build
cmake --build _build
```

## 第一个模型

### 全连接网络 (FNN)

```cpp
#include "CorePP.h"
#include "Model.h"
using namespace CoreNNSpace;

// 4输入 → 8 → 16 → 8 → 4输出，Sigmoid 激活
auto net = nn::FNN({4, 8, 16, 8, 4}, nn::Sigmoid);

// 输入是列向量 (input_size, 1)
Matrix<float> x(4, 1);  x << 3 << 4 << 2 << 1;
Matrix<float> y(4, 1);  y << 1 << 0 << 0 << 0;

// 训练
nn::Trainer(net, nn::MSE, Optim(net.getParams(), Adam, 0.01f))
    .fit(x, y, 300);
```

### 卷积网络 (CNN)

```cpp
// 3通道输入 → 4 → 4 通道卷积，3×3 卷积核
auto net = nn::CNN({3, 4, 4}, 3, nn::Sigmoid, 4);

// 输入是 3D tensor (H, W, C)
Matrix<float> x(8, 8, 3);
InitMatrixFunc(x, Uniform, {.a = -0.5, .b = 0.5});
Matrix<float> y(4, 1); y << 1 << 0 << 0 << 0;

nn::Trainer(net, nn::MSE, Optim(net.getParams(), Adam, 0.002f))
    .fit(x, y, 500);
```

### 循环网络 (RNN)

```cpp
RNN rnn(2, 4);  // input=2, hidden=4

// 输入: 每行一个时间步 (seq_len, input_size)
Matrix<float> x(5, 2);
x << 0.1 << 0.2 << 0.3 << 0.1 << 0.5 << 0.4 << 0.2 << 0.3 << 0.1 << 0.5;

Matrix<float> y(5, 4); y << 0.3 << 0.5 << 0.2 << 0.1 << 0;

nn::Trainer(rnn, nn::MSE, Optim({&rnn}, Adam, 0.005f))
    .fit(x, y, 800);
```

### LSTM 长短期记忆

```cpp
LSTM lstm(1, 4);  // input=1, hidden=4

// 长期记忆测试: 第一步输入 1，最后一步输出 1
Matrix<float> x(10, 1); x << 1 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0;
Matrix<float> y(10, 4);
y.at(9, 0) = 1; y.at(9, 1) = 1;

nn::Trainer(lstm, nn::MSE, Optim({&lstm}, Adam, 0.01f))
    .fit(x, y, 1000);
```

## 运行测试

```bash
./_build/tests/test_fnn
./_build/tests/test_cnn
./_build/tests/test_rnn
./_build/tests/test_lstm
./_build/tests/test_gradcheck
```
