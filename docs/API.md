# API 参考

## 命名空间

```cpp
using namespace CoreNNSpace;              // 所有类 (Matrix, Linear, Optim, ...)
using namespace CoreNNSpace::Activation;  // 激活函数 (ReLU, Sigmoid, ...)
using namespace CoreNNSpace::nn;          // 高层 API (FNN, CNN, Trainer)
```

---

## 高层 API (`Model.h`)

一行创建网络，自动组装层。

### nn::FNN — 全连接网络

```cpp
// nn::FNN({输入层, 隐藏层1, ..., 输出层}, 激活函数工厂)
// 参数1: initializer_list<int> — 每层的神经元数量
// 参数2: Module<float>* (*)() — 激活函数工厂函数指针
// 返回: Sequence — 可以直接 forward/backward

auto net = nn::FNN(
    {4, 8, 16, 8, 4},   // 层大小: 输入4 → 隐藏8 → 隐藏16 → 隐藏8 → 输出4
    nn::Sigmoid           // 每两层之间插入 Sigmoid 激活
);
// 等价于手写:
//   Sequence({new Linear(4,8), new Sigmoid(), new Linear(8,16), new Sigmoid(),
//             new Linear(16,8), new Sigmoid(), new Linear(8,4)})
```

### nn::CNN — 卷积网络

```cpp
// nn::CNN({输入通道, 中间通道..., 最后通道}, 卷积核大小, 激活函数, 分类数)
// 参数1: initializer_list<int> — 每层的通道数 (第一项是输入通道数)
// 参数2: int kernel — 卷积核大小 (默认3)
// 参数3: 激活函数工厂
// 参数4: int output_classes — Linear 输出类别数 (默认4)
// 自动追加 Flatten + Linear(output_shape, output_classes)

auto net = nn::CNN(
    {3, 4, 2},   // 通道: 3→4→2, 就是 Conv(3→4) + Conv(4→2)
    3,           // 3×3 卷积核
    nn::Sigmoid, // 每层 Conv 后接 Sigmoid
    4            // 最后 Linear(*, 4) 输出4类
);
// 等价于:
//   Conv2d(3,4,k3)→Sigmoid→Conv2d(4,2,k3)→Sigmoid→Flatten→Linear(?,4)
// 其中 ? = 自动计算的展平尺寸
```

### nn::Trainer — 训练循环

```cpp
// Trainer<ModelType>(模型引用, 损失函数类型, 优化器)
// 泛型: 接受 Sequence&、RNN&、LSTM& 等任意 Module 子类
// 方法 fit(输入, 目标, 轮数, 回调) — 执行训练循环

nn::Trainer trainer(
    net,                              // Sequence& — 你的网络
    nn::MSE,                          // LossType — MSE 或 CrossEntropy
    Optim(net.getParams(), Adam, 0.01f) // Optim — 优化器 (lr=0.01)
);

// 单样本训练: 同一个输入反复训练
trainer.fit(
    x,        // Matrix<float>& — 输入数据
    y,        // Matrix<float>& — 目标数据
    300,      // int — 训练轮数
    [](int epoch, float loss) {       // 回调: 每50轮调用一次
        printf("epoch %d: %.6f\n", epoch, loss);
    }
);

// 多样本训练: 两个样本交替
vector<pair<Matrix<float>*, Matrix<float>*>> samples = {{&x1,&y1}, {&x2,&y2}};
trainer.fit(samples, 500, callback);
```

### 优化器简写 + 损失简写

```cpp
nn::SGD(0.01f)              // → Optim({}, SGD, 0.01f)   注意: 不含 params
nn::Adam(0.001f)            // → Optim({}, Adam, 0.001f)
nn::RMSProp(0.01f)          // → Optim({}, RMSProp, 0.01f)

nn::MSE                     // LossType 枚举值
nn::CrossEntropy            // LossType 枚举值 (与 SoftMax 配合)
```

---

## Matrix\<T\> — 矩阵类

核心数据结构，2D (向量/矩阵) 或 3D (图像 tensor)。

```cpp
// ======== 构造 ========

Matrix<float> m(4, 1);                 // 2D: 4行×1列 (列向量)
Matrix<float> m(8, 8, 3);              // 3D: 8×8×3 (H×W×C, 图像)
Matrix<float> m(4, 1);
m << 1.0 << 2.0 << 3.0 << 4.0;        // 流式赋值, 逐行填充

// ======== 元素访问 ========

m.at(i, j)                             // 2D 读写: 返回 float&
m.at(i, j, ch)                         // 3D 读写: i行 j列 ch通道
m(i, j)                                // 2D 只读: 返回 float (const可用)
m.data_ptr()                           // 原始指针 float* (供 CUDA/GEMM 用)
const float* p = m.data_ptr();         // 只读指针

// ======== 矩阵运算 (全部 AVX2 SIMD 加速) ========

m1 * m2                                // 矩阵乘法: (M,K)×(K,N)→(M,N)
                                       //   内部: GotoBLAS 6×16 微内核
m1 + m2                                // 逐元素加: shape 必须完全一致
m1 - m2                                // 逐元素减
m1 & m2                                // 逐元素乘 (Hadamard 积)
m * 0.5f                               // 标量乘: 每个元素 ×0.5
m / 2.0f                               // 标量除
m.Translate()                           // 转置: (R,C)→(C,R), 返回新矩阵
m.sqrt()                               // 逐元素 sqrt

// ======== 显示 / 调试 ========

m.Analysis("图层名称");                 // 格式化打印:
                                       //   +=====+
                                       //   | 图层名称 [4x1]
                                       //   +=====+
                                       // 0 |+ 1.0000 |
                                       // 1 |- 0.0140 |
                                       //   +---------+
                                       // ±符号: +正数 -负数 ~接近零
                                       // 自适应: 小数→科学计数, 大数→少位

// ======== GPU 搬运 ========

m.device()                             // 返回 Device::CPU 或 Device::GPU
m.cuda()                               // CPU → GPU (需要 CUDA)
m.cpu()                                // GPU → CPU

// ======== 初始化 ========

Matrix<float> m(4, 1);                 // 默认: 未初始化 (零填充)
InitMatrixFunc(m, XavierUniform);      // Xavier 均匀分布初始化
InitMatrixFunc(m, HeNormal);           // He 正态分布初始化
InitMatrixFunc(m, Uniform, {.a=-1, .b=1}); // 均匀分布 [-1,1]
```

---

## 层

### 基类接口

所有层继承 `Module<float>`，拥有以下虚方法:

```cpp
layer.forward(Matrix<float>& input)     // 前向传播, 返回输出
layer.backward(Matrix<float>& grad)     // 反向传播, 返回输入梯度
layer.getParams()                       // → vector<Matrix<float>*> 权重列表
layer.getAllGrads()                     // → vector<Matrix<float>*> 梯度列表
layer.CleanGard()                       // 清理缓存的中间值 (optimizer step 后调用)
layer.cuda() / layer.cpu()              // GPU / CPU 搬运
layer.eval()  / layer.train()           // 推理模式 / 训练模式
```

### Linear — 全连接

```cpp
Linear(int in_features, int out_features, InitMode = XavierUniform)
//     ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^  输入维度  输出维度
// y = W·x + b    W: (out, in)   b: (out, 1)
// 输入: (in_features, 1) 列向量
// 输出: (out_features, 1) 列向量
```

### Conv2d — 2D 卷积

```cpp
Conv2d(int in_ch, int out_ch, int kernel=3, int stride=1, int padding=0,
//     ^^^^^^^^^  ^^^^^^^^^^  ^^^^^^^^^^^^  ^^^^^^^^^^^^  ^^^^^^^^^^^^^
//     输入通道数  输出通道数  卷积核大小    步长          填充
       InitMode = XavierUniform)

// 输入: (H, W, in_ch)    3D tensor
// 输出: (H', W', out_ch)  其中 H'=(H-k+2p)/s+1
// 内部: im2col + GEMM, weight_2d 缓存懒更新
```

### ConvTranspose2d — 转置卷积

```cpp
ConvTranspose2d(int in_channels, int out_channels, int kernel=3,
                int stride=2, int padding=1,
                InitMode mode=XavierUniform, double mu=0.0, double sigma=1.0)
// 输出尺寸: H_out = (H_in-1)*stride - 2*padding + kernel
// 本质: Conv2d 反向传播的前向版本 — W^T 乘输入 + col2im
// 用途: U-Net 解码器、GAN 生成器的上采样
```

### MaxPool2d — 最大池化

```cpp
MaxPool2d(int pool_size=2, int stride=2)
//        ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^   池化窗口 / 步长
// 输入: (H, W, C) → 输出: (H/2, W/2, C)
```

### AvgPool2d — 平均池化

```cpp
AvgPool2d(int pool_size=2, int stride=2)
// 反向传播: 梯度均匀分配给池化窗口内所有位置 (grad/ (pool²))
// 用途: ResNet 分类器前池化、DenseNet 过渡层
```

### AdaptiveAvgPool2d — 自适应平均池化

```cpp
AdaptiveAvgPool2d(int output_size=1)           // 正方形输出
AdaptiveAvgPool2d(int output_h, int output_w)  // 矩形输出
// 输出固定尺寸, 与输入 H,W 无关 — 全局平均池化 / 任意尺寸→固定尺寸
// 用途: 任意分辨率 CNN → 固定分类向量
```

### RNN — 循环网络

```cpp
RNN(int input_size, int hidden_size)
//  ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^   输入特征数 / 隐藏状态维度
// h_t = tanh(W_ih·x_t + W_hh·h_{t-1} + b_h)
// 输入: (seq_len, input_size)   每行一个时间步
// 输出: (seq_len, hidden_size)  每行对应 h_t
// 反向: BPTT, 梯度裁剪阈值 10
```

### LSTM — 长短期记忆

```cpp
LSTM(int input_size, int hidden_size)
// 4 门合并 GEMM: W(4H, I+H) × [x;h] + b(4H)
// 遗忘门 bias=+2 (默认记住), 输入门 bias=-2 (默认关闭)
// 输入/输出 shape 同 RNN
```

### GRU — 门控循环

```cpp
GRU(int input_size, int hidden_size)
// 2 门 (更新门+重置门), 参数比 LSTM 少 25%
// 输入/输出 shape 同 RNN
```

### BatchNorm1d — 批归一化

```cpp
BatchNorm1d(int features)
//          ^^^^^^^^^^^^   特征数
// 输入: (batch_size, features) — 每行一个样本
// 训练: 用当前 batch 的 mean/var
// 推理: 用 running mean/var (动量 0.9)
```

### LayerNorm — 层归一化

```cpp
LayerNorm(int features)
// 按样本归一化 (每行独立), 无 running stats
// 适合 RNN / Transformer
```

### BatchNorm2d — 批归一化 (图像)

```cpp
BatchNorm2d(int channels)
// 输入: (H, W, C) — 在 H×W 空间维度上归一化, per-channel
// 训练: 用当前 batch 的 per-channel mean/var
// 推理: 用 running mean/var (动量 0.9)
// 用途: CNN 训练 — 加速收敛、允许更高学习率
```

### GroupNorm — 分组归一化

```cpp
GroupNorm(int num_groups, int channels)
//         ^^^^^^^^^^^^^  ^^^^^^^^^^^^^   分组数 (通常 32) / 总通道数
// 输入: (H, W, C) — 在 (H,W,C/G) 维度上归一化
// 无 running stats, batch=1 也能正常工作
// 用途: 目标检测 (Faster R-CNN)、分割 (Mask R-CNN)、GAN (StyleGAN)
```

### Dropout — 随机失活

```cpp
Dropout(float rate=0.5f)
//       ^^^^^^^^^^^^^^   置零概率
// 训练: 随机将 rate 比例元素置零, 剩余 ×(1/(1-rate))
// 推理: 直通 (不做任何操作)
```

### Embedding — 词嵌入

```cpp
Embedding(int vocab_size, int embed_dim)
//        ^^^^^^^^^^^^^  ^^^^^^^^^^^^^   词汇表大小 / 嵌入维度
// 输入: (seq_len, 1) 整数索引
// 输出: (seq_len, embed_dim) 查表结果
```

### MultiHeadAttention — 多头注意力

```cpp
MultiHeadAttention(int d_model, int num_heads=8, bool causal=false)
//                 ^^^^^^^^^^^  ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^
//                 模型总维度      注意力头数         因果掩码 (decoder)
// d_model 必须被 num_heads 整除
// 输入/输出: (seq_len, d_model) — 自注意力保持 shape 不变
// 8 个参数矩阵: W_q, W_k, W_v, W_o + 4 bias
// 用途: Transformer 编码器/解码器核心
```

### Residual — 残差连接

```cpp
Residual(Module<float>* sublayer)
//       ^^^^^^^^^^^^^^^^^^^^^^^   被包装的子层 (拥有所有权)
// y = sublayer(x) + x
// 要求: sublayer 的输入输出 shape 相同
```

### ResNetBlock — 残差块

```cpp
ResNetBlock(int channels, bool use_bn=true)
//          ^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^   通道数 / 是否使用 BN
// Conv3×3(same)→BN?→ReLU→Conv3×3(same)→BN? + skip → ReLU
```

### GradientClipping — 梯度裁剪

```cpp
// 按范数裁剪: 保持梯度方向, 限制最大幅度
GradientClipping::clip_by_norm(modules, max_norm);
float norm = GradientClipping::clip_by_norm(modules, max_norm, true); // 返回范数

// 按值裁剪: 逐元素 clamp 到 [min_val, max_val]
GradientClipping::clip_by_value(modules, min_val, max_val);

// 单矩阵裁剪 (直接操作梯度张量)
GradientClipping::clip_by_norm(grad_matrix, max_norm);
GradientClipping::clip_by_value(grad_matrix, min_val, max_val);

// 用途: RNN/LSTM/GRU/Transformer 训练必需品, 防止梯度爆炸
```

---

## 激活函数

在 `CoreNNSpace::Activation` 子命名空间。全部使用方式相同:

```cpp
new Activation::ReLU()       // ReLU:     max(0, x), 导数: 0/1
new Activation::LeakyReLU(α) // LeakyReLU: x>0?x:αx, 默认 α=0.01
new Activation::Sigmoid()    // Sigmoid:  1/(1+e^{-x}), 裁剪 |x|>88
new Activation::Tanh()       // Tanh:     tanh(x)
new Activation::SoftMax()    // SoftMax:  e^x/Σe^x, O(N) 高效反向
new Activation::ELU(α)       // ELU:      x>0?x:α(e^x-1), 默认 α=1
new Activation::SELU()       // SELU:     自归一化 ELU
new Activation::Softplus()   // Softplus: ln(1+e^x)
new Activation::Mish()       // Mish:     x·tanh(softplus(x))
new Activation::GELU()       // GELU:     0.5x(1+tanh(√(2/π)(x+0.0447x³)))
new Activation::Swish()      // Swish:    x·sigmoid(x)
```

**实现模式**: 所有激活函数 forward 中存储 derivative 到 `gard`，backward 中执行 `grad ⊙ derivative`。

---

## 损失函数

```cpp
// MSE — 均方误差 (回归)
float mse_loss(pred, target)              // Σ(p-t)² / n
Matrix<float> mse_loss_grad(pred, target) // 梯度: p - t

// MAE (L1) — 平均绝对误差 (鲁棒回归)
float l1_loss(pred, target)               // Σ|p-t| / n
Matrix<float> l1_loss_grad(pred, target)  // 梯度: sign(p-t)

// Smooth L1 (Huber) — 混合损失
float smooth_l1_loss(pred, target, β=1)             // |d|<β→d²/2β else |d|-β/2
Matrix<float> smooth_l1_loss_grad(pred, target, β=1)

// Cross Entropy — 多分类 (输入 logits, 内部算 softmax)
Matrix<float> cross_entropy_loss_grad(logits, target)// softmax(logits)-target

// Binary Cross Entropy — 二分类
float bce_loss(pred, target)                         // -[t·log(p)+(1-t)·log(1-p)]
Matrix<float> bce_loss_grad(pred, target)

// KL Divergence — 分布差异
float kl_loss(log_probs, target_probs)  // Σ t·(log(t)-log(p)) / n
```

---

## 参数初始化

```cpp
enum InitMode {
    Zeros,          // 全 0
    Ones,           // 全 1
    Constant,       // 全 value
    Uniform,        // Uniform(a, b)         默认 [-1, 1]
    Gaussian,       // N(mu, sigma²)        默认 N(0, 1)
    XavierUniform,  // Uniform(-√6/(fan_in+fan_out), √6/(fan_in+fan_out))
    HeNormal,       // N(0, √2/fan_in)      适合 ReLU
    Identity,       // 单位矩阵 (需方阵)
    Orthogonal      // 正交矩阵 (Gram-Schmidt)
};

InitMatrixFunc(matrix, mode);                              // 默认参数
InitMatrixFunc(matrix, Uniform, {.a=-1, .b=1});            // 自定义范围
InitMatrixFunc(matrix, Gaussian, {.mu=0, .sigma=0.1});     // 自定义分布
InitMatrixFunc(matrix, XavierUniform);                      // 自动计算范围
```

---

## 优化器

```cpp
Optim optim(
    params,     // vector<Module<float>*> — 来自 seq.getParams()
    SGD,        // Optimizer 枚举: SGD/Momentum/Adam/AdamW/RMSProp/Adagrad/
                //                Adadelta/NAdam/RAdam
    0.01f       // float learn_rate — 学习率
);

optim.step(loss_gradient);   // 一步更新: backward + 参数更新 + CleanGard
                             // loss_gradient = mse_loss_grad(out, target)
```

**枚举值**:

| 枚举 | 全称 | 超参数 |
|------|------|--------|
| SGD | 随机梯度下降 | lr |
| Momentum | 动量法 | lr, β=0.9 |
| Adam | 自适应矩估计 | lr, β1=0.9, β2=0.999 |
| AdamW | 解耦权重衰减 | lr, wd=0.01 |
| RMSProp | 均方根传播 | lr, β=0.9 |
| Adagrad | 自适应梯度 | lr |
| Adadelta | 自适应学习率 | ρ=0.9 |
| NAdam | Nesterov Adam | lr, β1=0.9, β2=0.999 |
| RAdam | Rectified Adam | lr, β1=0.9, β2=0.999 |

---

## 学习率调度

```cpp
// 余弦退火
CosineLR sched(
    0.01f,   // lr_max — 初始学习率
    0.0001f, // lr_min — 最低学习率
    500,     // T_max — 周期 (epochs)
    [&](float new_lr) {
        optim = Optim(params, SGD, new_lr);  // setter — 每次更新回调
    }
);
sched.step();  // 每 epoch 调用

// 阶梯衰减
StepLR sched(
    0.01f,   // lr — 初始学习率
    50,      // step_size — 衰减间隔 (epochs)
    0.5f,    // gamma — 衰减系数
    [&](float new_lr) { /* setter */ }
);
```

---

## Sequence — 顺序容器

```cpp
Sequence seq({
    new Linear(4, 8),            // 第0层: 输入 4→8
    new Activation::Sigmoid(),   // 第1层: Sigmoid 激活
    new Linear(8, 4),            // 第2层: 8→4 输出
});

seq.forward(input);              // input → layer0 → layer1 → layer2 → output
seq.getParams();                 // 收集所有子层的 getParams()
seq.cuda();                      // 递归搬运所有子层到 GPU
seq.eval();                      // 递归设置 eval 模式
```

---

## DataLoader — 数据加载

### 抽象基类

```cpp
class DataLoader {
public:
    enum DataSplit { Train, Test, Val };

    virtual int  num_samples() const = 0;       // 当前 split 的样本数
    virtual bool next_batch(                     // 获取下一批数据
        Matrix<float>& x,                        // [out] 输入批次
        Matrix<float>& y) = 0;                   // [out] 目标批次
    // 返回 true=还有数据, false=epoch结束(自动reset)

    virtual void reset() {}                     // 重置迭代器
    virtual void set_split(DataSplit) {}        // 切换 Train/Test/Val

    void train() { set_split(Train); }          // 快捷切换
    void test()  { set_split(Test);  }
    void val()   { set_split(Val);   }
};
```

### 数据格式约定

| 网络类型 | 输入 shape | 目标 shape |
|----------|-----------|-----------|
| FNN (Linear) | `(features, 1)` 列向量 | `(classes, 1)` |
| CNN (Conv2d) | `(H, W, C)` 3D tensor | `(classes, 1)` |
| RNN/LSTM/GRU | `(seq_len, features)` 每行一步 | `(seq_len, hidden)` |

### 内置 Loader

```cpp
// 单样本: 只含一个 (x,y) 对, 每 epoch 重复 batch_size 次
SingleSampleLoader loader(x, y, batch_size=1);

// 内存: 从 vector 加载, 支持 train/test/val 划分
InMemoryLoader loader(batch_size=32, shuffle=true);
loader.add_sample(x1, y1);          // 逐个添加样本
loader.add_sample(x2, y2);
loader.split(0.7f, 0.2f, 0.1f);    // 70% train, 20% test, 10% val
loader.train();                      // 切换到训练集
loader.num_samples();                // → 训练集样本数
```

### 自定义 DataLoader

```cpp
class MyLoader : public DataLoader {
    // 你的数据源 (文件、数据库、网络...)
    vector<Matrix<float>> inputs_, targets_;
    int cursor_ = 0;

public:
    // 必须实现
    int num_samples() const override { return (int)inputs_.size(); }
    void reset() override { cursor_ = 0; }
    bool next_batch(Matrix<float>& x, Matrix<float>& y) override {
        if (cursor_ >= (int)inputs_.size()) { cursor_ = 0; return false; }
        x = inputs_[cursor_];       // 填充输出参数
        y = targets_[cursor_];
        cursor_++;
        return true;                // 还有数据
    }

    // 可选: 实现 set_split 切换训练/测试集
};
```

### Trainer 兼容

```cpp
MyLoader loader;
// ... 准备数据 ...

nn::Trainer(net, nn::MSE, Optim(net.getParams(), Adam, 0.01f))
    .fit(loader, 100, [](int e, float loss) {
        printf("epoch %d: %.4f\n", e, loss);
    });
// Trainer 内部自动调用 loader.reset() + while(loader.next_batch()) {...}
```

---

## 梯度检验

```cpp
#include "Autograd.h"

// numerical_gradient: 中心有限差分法计算数值梯度
//   f: function< float(const Matrix<float>&) > — 标量损失函数
//   x: 输入点
//   eps: 扰动大小 (默认 1e-5)
Matrix<float> ng = numerical_gradient<float>(loss_fn, x, 1e-4f);
// 每个元素: (f(x+ε) - f(x-ε)) / (2ε)

// gradcheck: 对比分析梯度 vs 数值梯度
//   threshold: 相对误差阈值 (默认 1e-3)
//   返回 true 如果所有元素误差 < threshold
bool ok = gradcheck(analytical_grad, numerical_grad, 1e-2f, "MyGrad");
```
