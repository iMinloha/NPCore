# 快速入门

## 环境要求

| 组件 | 最低版本 | 说明 |
|------|----------|------|
| 编译器 | GCC 12+ 或 MSVC 2022 | MinGW-w64 推荐 |
| CMake | 3.18+ | |
| CUDA Toolkit | 11.0+ (可选) | GPU 加速 |
| VS Build Tools | 2022 (可选) | CUDA host 编译器 |

## 构建

```bash
# CPU 版本
cmake -G "MinGW Makefiles" -B _build
cmake --build _build

# CUDA 版本
cmake -G "MinGW Makefiles" -B _build -DCOREPP_ENABLE_CUDA=ON
cmake --build _build

# 动态库
cmake -G "MinGW Makefiles" -B _build -DBUILD_SHARED_LIBS=ON
cmake --build _build
```

## 第一个模型

### FNN 全连接网络

```cpp
#include "CorePP.h"
using namespace CoreNNSpace;

// 4输入 → 8 → 16 → 8 → 4输出
Sequence net({
    new Linear(4, 8),   new Activation::Sigmoid(),
    new Linear(8, 16),  new Activation::Sigmoid(),
    new Linear(16, 8),  new Activation::Sigmoid(),
    new Linear(8, 4, XavierUniform),
});

Matrix<float> x(4,1); x << 3 << 4 << 2 << 1;
Matrix<float> y(4,1); y << 1 << 0 << 0 << 0;

Optim optim(net.getParams(), Adam, 0.01f);
for (int e = 0; e < 300; e++) {
    auto out = net.forward(x);
    optim.step(out - y);  // MSE 梯度
}
```

### 使用高层 API

```cpp
#include "CorePP.h"
#include "Model.h"

auto net = nn::FNN({4, 8, 4}, nn::Sigmoid);

nn::Trainer(net, nn::MSE, Optim(net.getParams(), Adam, 0.01f))
    .fit(x, y, 300, [](int e, float loss) {
        printf("epoch %d: %.6f\n", e, loss);
    });
```

### CNN 卷积网络

```cpp
// 1通道 → Conv→ReLU→Pool → Conv→ReLU → Flatten → 2分类
Sequence net({
    new Conv2d(1, 4, 3, 1, 0),
    new Activation::ReLU(),
    new MaxPool2d(2, 2),
    new Conv2d(4, 4, 3, 1, 0),
    new Activation::ReLU(),
    new Flatten(),
    new Linear(4, 2),
});

Matrix<float> x(8, 8, 1);  // 3D tensor (H,W,C)
InitMatrixFunc(x, Uniform, {.a = -0.5, .b = 0.5});
```

### RNN 循环网络

```cpp
RNN rnn(2, 4);  // input=2, hidden=4

// 输入: (seq_len, features) — 每行一个时间步
Matrix<float> x(5, 2);
x << 0.1 << 0.2 << 0.3 << 0.1 << 0.5 << 0.4 << 0.2 << 0.3 << 0.1 << 0.5;

Optim optim({&rnn}, Adam, 0.005f);
for (int e = 0; e < 500; e++) {
    auto out = rnn.forward(x);
    optim.step(out - y);
}
```

### LSTM 长短期记忆

```cpp
LSTM lstm(1, 4);

// 10步序列，第一步输入1，最后一步输出1
Matrix<float> x(10, 1);
x << 1 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0;

Matrix<float> y(10, 4);
y.at(9, 0) = 1; y.at(9, 1) = 1;  // 目标: 最后一步输出1
```

### ResNet 残差块

```cpp
// 内联自定义 ResNet 块
class MyResBlock : public Module<float> {
    Conv2d *c1, *c2;
    Activation::ReLU *r1, *r2;
    Matrix<float>* skip = nullptr;
public:
    MyResBlock(int ch) {
        c1 = new Conv2d(ch, ch, 3, 1, 1);
        c2 = new Conv2d(ch, ch, 3, 1, 1);
        r1 = new Activation::ReLU(); r2 = new Activation::ReLU();
    }
    Matrix<float> forward(Matrix<float>& x) override {
        delete skip; skip = new Matrix<float>(x);  // 保存输入给 skip
        auto o = r2->forward(c2->forward(r1->forward(c1->forward(x))));
        int n = o.row * o.col * o.channel;
        for (int i = 0; i < n; ++i) o.data_ptr()[i] += skip->data_ptr()[i];
        return o;  // y = F(x) + x
    }
    // ... backward/getParams/getAllGrads/CleanGard 参见 CustomModel.md
};
```

### GPU 加速

```cpp
net.cuda();                        // 所有权重搬到 GPU
auto out = net.forward(x);         // 全程 GPU 运算
out.cpu();                         // 搬回 CPU 查看
out.Analysis("GPU result");
```

### 梯度检验

```cpp
#include "Autograd.h"

auto loss_fn = [&](const Matrix<float>& x) -> float {
    auto out = net.forward(x);
    return mse_loss(out, y);
};

Matrix<float> num_grad = numerical_gradient<float>(loss_fn, x, 1e-4f);
Matrix<float> ana_grad = /* 分析梯度 */;
bool ok = gradcheck(ana_grad, num_grad, 1e-2f, "MyGrad");
```
