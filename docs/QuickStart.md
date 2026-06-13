# 快速入门

## 环境要求

| 组件 | 最低版本 | 说明 |
|------|----------|------|
| 编译器 | GCC 12+ 或 MSVC 2022 | MinGW-w64 推荐，MSVC 需配合 nvcc |
| CMake | 3.18+ | 构建系统 |
| OpenMP | 系统自带 | 自动检测，用于 CPU 并行 |
| CUDA Toolkit | 11.0+ (可选) | GPU 加速 |
| VS Build Tools | 2022 (可选) | CUDA 编译需要 MSVC 作为 host compiler |

## 安装

```bash
# 1. 克隆项目
git clone <url> && cd NPCore

# 2. CPU 编译 (MinGW)
cmake -G "MinGW Makefiles" -B _build   # -G 指定生成器, -B 指定构建目录
cmake --build _build                    # 编译, 输出在 _build/lib/ 和 _build/bin/

# 3. 运行示例
./_build/examples/example_fnn           # 全连接网络示例
```

## 第一个模型: 全连接网络

```cpp
#include "NPCore.h"                     // 所有模块的入口头文件
using namespace NPCore;            // 引入核心命名空间

int main() {
    // ---- 第1步: 构建网络 ----
    // Sequence 是层的顺序容器，forward 时逐层传递
    Sequence net({
        new Linear(4, 8),               // 全连接: 输入4维 → 输出8维
                                        //   权重 (8×4), 偏置 (8×1)
        new Activation::Sigmoid(),      // Sigmoid 激活: 1/(1+e^{-x})
        new Linear(8, 16),              // 第二层: 8→16
        new Activation::Sigmoid(),
        new Linear(16, 8),              // 第三层: 16→8
        new Activation::Sigmoid(),
        new Linear(8, 4, XavierUniform),// 输出层: 8→4, Xavier 权重初始化
    });

    // ---- 第2步: 准备数据 ----
    // Matrix(rows, cols) - 2D 矩阵, 列向量用 (N, 1)
    Matrix<float> x(4, 1);              // 输入: 4行1列 (4维特征向量)
    x << 3 << 4 << 2 << 1;             // 流式赋值: 逐行填充
    Matrix<float> y(4, 1);              // 目标: one-hot 编码
    y << 1 << 0 << 0 << 0;             // 第0类为1, 其余为0

    // ---- 第3步: 检查初始状态 ----
    auto out = net.forward(x);          // 前向传播: 输入经过所有层
    out.Analysis("Initial");            // 格式化打印矩阵 (±符号, 自适应精度)
    cout << "MSE: " << mse_loss(out, y) << endl;  // 均方误差

    // ---- 第4步: 训练 ----
    // Optim(参数列表, 优化器类型, 学习率)
    Optim optim(net.getParams(),        // 收集所有层的权重指针
                RMSProp,                // RMSProp 优化器
                0.01f);                 // 学习率

    for (int e = 0; e < 300; e++) {
        out = net.forward(x);           // 前向
        optim.step(out - y);            // 一步: backward + 参数更新
                                        //   out-y 即为 MSE 的梯度
        if (e % 100 == 0)
            cout << "  epoch " << e << ": " << mse_loss(out, y) << endl;
    }

    // ---- 第5步: 结果 ----
    out = net.forward(x);
    out.Analysis("Final");              // 打印最终预测
    y.Analysis("Target");               // 打印目标值对比
}
```

## 更多模型

### CNN 卷积网络

```cpp
// Conv(1→4,k3) → ReLU → MaxPool(2) → Conv(4→4,k3) → ReLU → Flatten → Linear(4→2)
Sequence net({
    new Conv2d(1, 4, 3, 1, 0),         // 1ch→4ch, 3×3核, stride=1, padding=0
    new Activation::ReLU(),             // ReLU: max(0,x)
    new MaxPool2d(2, 2),                // 2×2 窗口, stride=2 → 尺寸减半
    new Conv2d(4, 4, 3, 1, 0),         // 4ch→4ch
    new Activation::ReLU(),
    new Flatten(),                      // 多维→1D: (H,W,C)→(H*W*C, 1)
    new Linear(4, 2),                   // 4→2 分类
});

// 输入是 3D tensor: (Height, Width, Channels)
Matrix<float> x(8, 8, 1);              // 8×8 单通道图像
InitMatrixFunc(x, Uniform, {.a=-0.5, .b=0.5});  // 随机初始化 [-0.5, 0.5]
Matrix<float> y(2, 1); y << 1 << 0;    // 二分类: 第0类
```

### RNN 序列预测

```cpp
// RNN 是独立层 (不在 Sequence 中, 因其输入输出格式特殊)
RNN rnn(2, 4);                          // 输入2维, 隐藏状态4维
                                        // h_t = tanh(W_ih·x_t + W_hh·h_{t-1} + b)

// 输入: (seq_len, features) - 每行是一个时间步
Matrix<float> x(5, 2);                  // 5个时间步, 每步2个特征
x << 0.1 << 0.2  << 0.3 << 0.1  << 0.5 << 0.4
  << 0.2 << 0.3  << 0.1 << 0.5;

Matrix<float> y(5, 4);                  // 目标: 每个时间步对应4维输出

// RNN 作为独立层使用时，Optim 需要显式传递 Module 指针
Optim optim({&rnn}, Adam, 0.005f);      // {&rnn} → vector<Module<float>*>
for (int e = 0; e < 500; e++) {
    auto out = rnn.forward(x);
    optim.step(out - y);
}
```

### LSTM 长期记忆

```cpp
LSTM lstm(1, 4);                        // 1维输入, 4维隐藏状态

// 经典任务: 第一步输入1, 10步后在最后一步输出1
Matrix<float> x(10, 1);                 // 10 个时间步
x << 1 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0;

Matrix<float> y(10, 4);                 // 目标: 第9步 (最后) 输出1
y.at(9, 0) = 1;                         //   只监督最后一步的 dim0
y.at(9, 1) = 1;                         //   其他位置默认为0

Optim optim({&lstm}, Adam, 0.01f);
for (int e = 0; e < 800; e++) {
    auto out = lstm.forward(x);
    optim.step(out - y);                // 梯度通过 10 个时间步反向传播
}
```

### ResNet 残差网络

```cpp
// 自定义 ResNet 块 (内联在文件中, 不需修改 NPCore 源码)
class MyResBlock : public Module<float> {
    Conv2d *c1, *c2;                    // 两个卷积层
    Activation::ReLU *r1, *r2;          // 两个激活
    Matrix<float>* skip = nullptr;      // 存储 skip 连接的输入
public:
    MyResBlock(int ch) {
        c1 = new Conv2d(ch, ch, 3, 1, 1);  // same padding: 输入输出尺寸相同
        c2 = new Conv2d(ch, ch, 3, 1, 1);
        r1 = new Activation::ReLU();
        r2 = new Activation::ReLU();
    }

    Matrix<float> forward(Matrix<float>& x) override {
        delete skip;
        skip = new Matrix<float>(x);    // 保存原始输入给 skip 连接
        auto o = c1->forward(x);
        o = r1->forward(o);
        o = c2->forward(o);
        int n = o.row * o.col * o.channel;
        for (int i = 0; i < n; ++i)
            o.data_ptr()[i] += skip->data_ptr()[i];  // skip: o = F(x) + x
        o = r2->forward(o);
        return o;
    }
    // ... backward/getParams/getAllGrads/CleanGard 参见 DevGuide.md
};

// 使用内联块构建网络
Sequence net({
    new Conv2d(1, 4, 3, 1, 1),         // 6x6x1 → 6x6x4
    new Activation::ReLU(),
    new MyResBlock(4),                  // 6x6x4 → 6x6x4 (skip connection)
    new MaxPool2d(2, 2),                // 6x6x4 → 3x3x4
    new Flatten(),                      // → 36
    new Linear(36, 2),                  // → 2 分类
});
```

### GPU 加速

```cpp
net.cuda();                             // 一行: 所有权重搬到 GPU
                                        //   内部递归调用所有子层的 cuda()
                                        //   Matrix::to(Device::GPU): cudaMalloc + memcpy H2D

auto out = net.forward(x);              // 前向全程在 GPU 上
                                        //   Matrix::operator* 检测双方都是 GPU
                                        //   → cuda_gemm_device (GPU 指针直接运算, 零拷贝)

out.cpu();                              // 搬回 CPU 查看结果
out.Analysis("GPU Result");             // 打印

net.cpu();                              // 搬回 CPU (不再需要 GPU)
```

### 梯度检验

```cpp
#include "Autograd.h"

// 1. 定义标量损失函数
auto loss_fn = [&net, &y](const Matrix<float>& x) -> float {
    auto out = net.forward(x);          // 前向 (注意: 会修改 net 内部状态!)
    return mse_loss(out, y);            // 返回标量损失
};

// 2. 数值梯度 (中心有限差分)
//    对每个元素: grad[i] = (f(x+ε) - f(x-ε)) / (2ε)
Matrix<float> num = numerical_gradient<float>(loss_fn, x, 1e-4f);

// 3. 分析梯度 (通过 backward)
auto out = net.forward(x);
Matrix<float> ana = net.backward(out - y);  // 或逐层 backward

// 4. 对比
bool ok = gradcheck(ana, num, 1e-2f, "MyGradient");
//  阈值 1e-2: 相对误差 < 1% 认为通过
//  输出: 每个元素的 ana vs num 对比, max error 报告
```
