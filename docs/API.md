# NPCore API Reference

## 1. Matrix — 核心数据结构

所有数据以 `Matrix<float>` 表示，**行主序 (row-major)** 存储。

```cpp
#include "NPCore.h"
using namespace NPCore;

// 构造
Matrix<float> x(3, 1);         // 3 行 1 列 (列向量, 特征)
Matrix<float> y(3, 2);         // 3 行 2 列
Matrix<float> img(28, 28, 1);  // 28×28 单通道图像 (H, W, C)

// 元素访问
x.at(0, 0) = 1.0f;             // 2D 索引
x.at(0, 0, 2) = 1.0f;          // 3D 索引 (带通道)
float v = x(0, 0);              // 只读访问
float* raw = x.data_ptr();      // 原始指针

// 填充
x << 1.0f << 2.0f << 3.0f;     // 逐元素填充 (按行)

// 运算
auto z = a * b;                 // 矩阵乘法
auto z = a + b;                 // 逐元素加法
auto z = a - b;                 // 逐元素减法
auto z = a & b;                 // 逐元素乘法 (Hadamard)
auto z = a / 2.0f;              // 标量除法
auto t = a.Translate();         // 转置

// 调试
x.Analysis("weights");          // ASCII 表格打印矩阵内容
```

### 形状约定

| 数据 | 形状 | 示例 |
|------|------|------|
| 特征向量 | `(features, 1)` | `Matrix<float> x(4, 1)` |
| 批量样本 | `(batch, features)` | 每行一个样本 |
| 时序输入 | `(seq_len, features)` | RNN/LSTM/GRU |
| 图像 | `(H, W, channels)` | Conv2d 输入 |
| 标签 (分类) | `(classes, 1)` one-hot | `[0,1,0]ᵗ` 表示第 2 类 |
| 标签 (回归) | `(dims, 1)` | 连续值 |

---

## 2. 数据加载器 — DataLoader

### 2.1 数据格式总览

每个 loader 的 `next_batch(x, y)` 输出两种矩阵：

| 参数 | 含义 | 要求 |
|------|------|------|
| `x` | 输入特征 | 形状取决于模型 |
| `y` | 目标/标签 | 形状匹配损失函数 |

### 2.2 SingleSampleLoader — 单样本重复

```cpp
Matrix<float> x(4, 1); x << 1 << 2 << 3 << 4;
Matrix<float> y(1, 1); y << 0.5f;
SingleSampleLoader loader(x, y, /*batch_size=*/10);
// next_batch 重复返回同一份 (x, y) 共 batch_size 次
```

| 输入要求 | 输出格式 |
|----------|----------|
| 两个 `Matrix<float>` (任意形状) | 原样返回 |

### 2.3 InMemoryLoader — 内存数据集

```cpp
InMemoryLoader loader(/*batch_size=*/32, /*shuffle=*/true);
loader.add_sample(x1, y1);
loader.add_sample(x2, y2);
// ...
loader.split(0.7f, 0.2f, 0.1f);  // 70% train, 20% test, 10% val
loader.train();                    // 切换到训练集
while (loader.next_batch(xb, yb)) { ... }  // 每批 32 个
```

| 输入要求 | 输出格式 |
|----------|----------|
| 每个样本独立的 `Matrix<float>` | `batch_size=1` 时逐样本；`>1` 时沿行堆叠为 `(batch, features)` |

### 2.4 CSVLoader — CSV 文件加载

```cpp
CSVLoader csv("data.csv", /*target_cols=*/1, /*has_header=*/false);
csv.load();  // 解析文件
csv.split(0.7f, 0.3f);
```

**CSV 格式约定**：末尾 `target_cols` 列是目标值，前面是特征值。

```
# 示例: 3 特征 + 1 目标
1.0,2.0,3.0,0.5
4.0,5.0,6.0,0.8
```

| 输入要求 | 输出格式 |
|----------|----------|
| CSV 文本，逗号分隔，纯数值 | `x=(features,1)` 列向量, `y=(targets,1)` |

### 2.5 ImageFolderLoader — 图像文件夹

```cpp
ImageFolderLoader img_loader("./images", [](const string& path) -> Matrix<float> {
    // 自定义解码器: 返回 (H, W, C) 矩阵
    // 值域建议归一化到 [0,1] 或 [-1,1]
    return your_decode(path);
}, /*batch_size=*/32);
img_loader.scan();  // 扫描子文件夹
img_loader.split(0.7f, 0.3f);
```

**目录结构**：
```
images/
  cat/    → 标签 0
    img001.jpg
    img002.jpg
  dog/    → 标签 1
    img001.jpg
```

| 输入要求 | 输出格式 |
|----------|----------|
| 子文件夹=类别名, 自定义解码器 | `x=(H,W,C)`, `y=(classes,1)` one-hot |

### 2.6 SequenceLoader — 变长序列

```cpp
SequenceLoader seq_loader(/*input_dim=*/3, /*output_dim=*/2, /*batch_size=*/16);
seq_loader.add_sequence(seq1_x, seq1_y);  // (seq_len, dim)
seq_loader.add_sequence(seq2_x, seq2_y);
seq_loader.split(0.7f, 0.3f);
```

| 输入要求 | 输出格式 |
|----------|----------|
| 每个序列 `(seq_len, feat_dim)` | 批次内 padding 到 `max_len`, `x=(max_len, feat*batch)`, `mask=(max_len, batch)` |

### 2.7 BatchStackLoader — 批包装器

```cpp
BatchStackLoader batched(&base_loader, /*batch_size=*/32, /*pad_seq=*/true);
batched.last_lengths();  // 变长序列 padding 后的实际长度
```

---

## 3. 层 (Layers) — 输入/输出格式

### 3.1 Linear (全连接)

```
y = W·x + b
```

| 参数 | 构造 | 输入 | 输出 |
|------|------|------|------|
| `in_features` | 输入维度 | `(in_features, 1)` | `(out_features, 1)` |
| `out_features` | 输出维度 | | |

```cpp
auto linear = new Linear(4, 8);           // 4→8
auto out = linear->forward(x);            // x: (4,1) → out: (8,1)
```

### 3.2 Conv2d (二维卷积)

```
im2col → GEMM → reshape + bias
```

| 参数 | 构造 | 输入 | 输出 |
|------|------|------|------|
| `in_channels` | 输入通道 | `(H, W, C_in)` | `(H_out, W_out, C_out)` |
| `out_channels` | 输出通道 | | |
| `kernel_size=3` | 卷积核大小 | | H_out = (H-K+2P)/S+1 |
| `stride=1` | 步长 | | |
| `padding=0` | 填充 | | |

```cpp
auto conv = new Conv2d(3, 16, 3, 1, 0);  // 3→16 通道, 3×3 核
auto out = conv->forward(img);            // (32,32,3)→(30,30,16)
```

### 3.3 ConvTranspose2d (转置卷积)

```
输出尺寸 = (H_in-1)*stride - 2*padding + kernel_size
```

用于上采样 (U-Net decoder, GAN generator)。

### 3.4 Pooling (池化)

| 类 | 输入 | 输出 |
|----|------|------|
| `MaxPool2d(pool=2, stride=2)` | `(H,W,C)` | `(H/2, W/2, C)` |
| `AvgPool2d(pool=2, stride=2)` | `(H,W,C)` | `(H/2, W/2, C)` |
| `AdaptiveAvgPool2d(h, w)` | `(H,W,C)` | `(h, w, C)` 固定 |

### 3.5 RNN / LSTM / GRU (循环层)

```
输入: (seq_len, input_size)   — 每行一个时间步
输出: (seq_len, hidden_size)  — 每行对应的隐状态
```

| 参数 | 含义 |
|------|------|
| `input_size` | 每个时间步的特征维度 |
| `hidden_size` | 隐状态维度 |

```cpp
auto lstm = new LSTM(/*input=*/3, /*hidden=*/16);
auto out = lstm->forward(seq);  // (100, 3) → (100, 16)
```

### 3.6 Normalization (归一化)

| 类 | 输入 | 说明 |
|----|------|------|
| `BatchNorm1d(features)` | `(N, features)` | 沿 batch 维度归一化 |
| `BatchNorm2d(channels)` | `(H, W, C)` | 沿空间维度逐通道归一化 |
| `LayerNorm(features)` | `(N, features)` | 逐样本归一化 (Transformer) |
| `GroupNorm(groups, channels)` | `(H, W, C)` | 通道分组归一化 |

### 3.7 激活函数

| 类 | 输出范围 | 特点 |
|----|---------|------|
| `ReLU` | `[0, +∞)` | 最常用 |
| `LeakyReLU(alpha=0.01)` | `(-∞, +∞)` | 缓解 dead neuron |
| `Sigmoid` | `(0, 1)` | 二分类输出层 |
| `Tanh` | `(-1, 1)` | RNN 隐层 |
| `SoftMax` | `[0,1]` 行和为 1 | 多分类输出层 |
| `GELU` | `(-0.17, +∞)` | Transformer |
| `Swish` | `(-0.28, +∞)` | EfficientNet |
| `ELU(alpha=1.0)` | `(-alpha, +∞)` | 平滑 ReLU |
| `SELU` | 自归一化 | 深层网络 |
| `Softplus` | `(0, +∞)` | 平滑 ReLU |
| `Mish` | `(-0.31, +∞)` | YOLOv4 |

### 3.8 其他层

| 类 | 用途 |
|----|------|
| `Flatten` | `(H,W,C) → (H*W*C, 1)`, Conv→Linear 转换 |
| `Dropout(rate=0.5)` | 训练时随机置零, 推理时恒等 |
| `Embedding(vocab, dim)` | 整数索引 → 稠密向量 |
| `MultiHeadAttention(d_model, heads)` | Transformer 自注意力 |
| `PositionalEncoding(d_model)` | 正弦/余弦位置编码，Transformer 必需 |

---

## 4. 模型组装

### 4.1 Sequence (顺序容器)

`Sequence` 继承 `Module<float>`，支持**嵌套**和**递归**参数管理。

```cpp
// 扁平用法
vector<Module<float>*> layers = {
    new Linear(4, 16), new Activation::ReLU(),
    new Linear(16, 8), new Activation::ReLU(),
    new Linear(8, 2),  new Activation::SoftMax()
};
Sequence model(layers);

// 嵌套用法 — 内层 Sequence 可以作为外层的一个 Module
auto* encoder = new Sequence({
    new Linear(64, 32), new Activation::ReLU(),
    new Linear(32, 16), new Activation::ReLU()
});
auto* decoder = new Sequence({
    new Linear(16, 32), new Activation::ReLU(),
    new Linear(32, 64), new Activation::Sigmoid()
});
Sequence autoencoder({encoder, decoder});
// autoencoder.modules() 递归展平所有叶子层
// autoencoder.getParams() 返回所有权重矩阵（递归展平）

// 工厂
auto model = nn::FNN({4, 16, 8, 2}, nn::ReLU);            // MLP
auto model = nn::CNN({3, 16, 32}, 3, nn::ReLU, 10);       // CNN
```

### 4.2 Sequence 方法对照

| 方法 | 返回 | 说明 |
|------|------|------|
| `getParams()` | `vector<Matrix<float>*>` | 所有权重矩阵（递归展平） |
| `modules()` | `vector<Module<float>*>` | 所有叶子层（用于 Optimizer） |
| `getLayers()` | `const vector<Module<float>*>&` | 直接子层引用 |
| `getAllGrads()` | `vector<Matrix<float>*>` | 所有权重梯度（递归展平） |

### 4.3 Concat (分支合并)

多分支并行 + 拼接，用于 Inception / U-Net 等结构。

```cpp
auto* branch1 = new Sequence({new Conv2d(1,4,3,1,1), new Activation::ReLU()});
auto* branch2 = new Sequence({new Conv2d(1,3,5,1,2), new Activation::ReLU()});
Concat inception({branch1, branch2});
// 输入 (H,W,C) → 分支并行 → 沿通道拼接 (H,W,7)
// 输入 (N,1) 2D → 分支并行 → 沿行拼接 (N1+N2,1)
```

### 4.4 Residual (残差连接)

```cpp
auto* block = new Residual(new Linear(64, 64));  // y = F(x) + x
```

### 4.5 ResNetBlock

```cpp
auto* resblock = new ResNetBlock(/*channels=*/64, /*use_bn=*/true);
// Conv3→BN→ReLU→Conv3→BN + skip →ReLU
```

---

## 5. 损失函数

```cpp
using namespace NPCore;

// 回归
float loss = mse_loss(pred, target);              // MSE
float loss = l1_loss(pred, target);               // MAE
float loss = smooth_l1_loss(pred, target, 1.0f);  // Huber

// 分类
float ce = cross_entropy_loss(logits, labels);    // 带 softmax 的交叉熵 (值)
float ce_b = bce_loss(pred, target);              // 二分类交叉熵

// 梯度
Matrix<float> grad = mse_loss_grad(pred, target);
Matrix<float> grad = l1_loss_grad(pred, target);
Matrix<float> grad = smooth_l1_loss_grad(pred, target, 1.0f);
Matrix<float> grad = cross_entropy_loss_grad(logits, labels);  // CE 梯度 (softmax 内置)
Matrix<float> grad = bce_loss_grad(pred, target);              // BCE 梯度

// KL 散度
float kl = kl_loss(log_pred, target_prob);
```

| 损失 | pred 形状要求 | target 形状要求 | 数据要求 |
|------|-------------|---------------|---------|
| `mse_loss` | 任意 | 相同形状 | 回归, 值域不限 |
| `l1_loss` | 任意 | 相同形状 | 回归, 对离群值鲁棒 |
| `smooth_l1_loss` | 任意 | 相同形状 | 回归, Huber |
| `cross_entropy_loss` | `(N, classes)` | `(N, classes)` | 分类, logits 输入 |
| `cross_entropy_loss_grad` | `(N, classes)` | `(N, classes)` | 分类, logits 输入, 内置 softmax |
| `bce_loss` | 任意 | 相同形状 | 二分类, pred ∈ (0,1) |
| `bce_loss_grad` | 任意 | 相同形状 | 二分类, 梯度已归一化 |

---

## 6. 优化器

```cpp
// 工厂函数 (使用默认超参数)
Optim sgd     = SGD(0.01f);        // SGD lr=0.01
Optim momentum = Momentum(0.01f);  // Momentum lr=0.01, β=0.9
Optim adam    = Adam(0.001f);      // Adam lr=0.001, β₁=0.9, β₂=0.999
Optim rms     = RMSProp(0.01f);    // RMSProp lr=0.01, β=0.99
Optim adagrad = Adagrad(0.01f);    // Adagrad lr=0.01
Optim nadam   = NAdam(0.001f);     // NAdam lr=0.001
Optim radam   = RAdam(0.001f);     // RAdam lr=0.001

// 直接构造 (自定义超参数)
Optim opt(Adam_step, seq.modules(), 0.001f);
```

### 可用优化器

| 工厂 | 底层函数 | 特点 |
|------|---------|------|
| `SGD(lr)` | `SGD_step` | 朴素梯度下降 |
| `Momentum(lr)` | `Momentum_step` | 带动量的 SGD |
| `Adam(lr)` | `Adam_step` | 自适应学习率 (推荐) |
| `RMSProp(lr)` | `RMSProp_step` | 适合 RNN |
| `Adagrad(lr)` | `Adagrad_step` | 稀疏特征 |
| `Adadelta(lr)` | `Adadelta_step` | 无需手动设 lr |
| `NAdam(lr)` | `NAdam_step` | Nesterov + Adam |
| `RAdam(lr)` | `RAdam_step` | 整流 Adam, 训练更稳定 |

### 其他优化器

```cpp
// AdamW (解耦权重衰减) — 独立类
AdamW adamw(seq.modules(), /*lr=*/0.001f, /*weight_decay=*/0.01f);
adamw.step(grad);

// 学习率调度
CosineLR sched(0.01f, 0.0001f, /*T_max=*/200, [&](float lr) { adamw.set_lr(lr); });
StepLR step_lr(0.01f, /*step_size=*/30, /*gamma=*/0.1f, setter);
for (int e = 0; e < epochs; e++) { sched.step(); ... }

// 梯度裁剪
GradientClipping::clip_by_norm(model.getLayers(), /*max_norm=*/10.0f);
```

### 训练模式

```cpp
auto model = nn::FNN({4, 8, 4}, nn::Sigmoid);
model.train();
Matrix<float> out = model.forward(x);
// 向后传播集成在 Optim 内部
Optim optim(Adam_step, model.getParams(), 0.001f);
optim.step(loss_grad(out, target, nn::MSE));
model.eval();
```

---

## 7. 高层 Trainer API

```cpp
// 方式 1: 单样本反复训练
auto net = nn::FNN({4, 8, 4}, nn::Sigmoid);
nn::Trainer trainer(net, nn::MSE, Adam(0.001f));
// Trainer 接受 Sequence&，自动将模型层参数注入 Optimizer
trainer.fit(input, target, /*epochs=*/200, [](int e, float loss) {
    printf("epoch %d: %.4f\n", e, loss);
});

// 方式 2: 批量样本
vector<pair<Matrix<float>*, Matrix<float>*>> samples = { ... };
trainer.fit(samples, 200);

// 方式 3: DataLoader
InMemoryLoader loader(32);
// ... 添加样本 ...
trainer.fit(loader, 50);

// 更换优化器
trainer.bind(Adam(0.0005f));  // 自动重新绑定模型参数
```

---

## 8. 数据预处理标准

### 归一化

| 场景 | 方法 | 值域 |
|------|------|------|
| 图像 (通用) | `x / 255.0f` | `[0, 1]` |
| 图像 (ImageNet 风格) | `(x/255 - mean) / std` | `~[-2, 2]` |
| 表格特征 | Z-score: `(x - μ) / σ` | 均值 0, 方差 1 |
| RNN 时序 | Min-Max 或 Z-score | 避免梯度爆炸 |

### 标签格式

| 任务 | y 格式 | 示例 |
|------|--------|------|
| 二分类 | `(1, 1)` 0 或 1 | `y << 1.0f;` |
| 多分类 | `(classes, 1)` one-hot | `y << 0 << 1 << 0;` (第 2 类) |
| 回归 | `(dims, 1)` 连续值 | `y << 0.5f;` |

---

## 9. 完整示例

### MLP 二分类

```cpp
#include "NPCore.h"
#include "Model.h"
using namespace NPCore;

int main() {
    // 数据: 100 个样本, 每个 2 特征
    InMemoryLoader loader(32);
    for (int i = 0; i < 100; i++) {
        float a = RandomGenerator::uniform<float>(-1, 1);
        float b = RandomGenerator::uniform<float>(-1, 1);
        Matrix<float> x(2, 1); x << a << b;
        Matrix<float> y(2, 1); y << (a+b>0?1.0f:0.0f) << (a+b>0?0.0f:1.0f);  // one-hot
        loader.add_sample(x, y);
    }
    loader.split(0.7f, 0.3f);

    // 模型
    auto net = nn::FNN({2, 16, 8, 2}, nn::ReLU);

    // 训练 (Trainer 自动管理参数 + optimizer)
    nn::Trainer trainer(net, nn::CrossEntropy, Adam(0.001f));
    trainer.fit(loader, 200, [](int e, float loss) {
        if (e % 20 == 0) printf("e%3d loss=%.4f\n", e, loss);
    });

    // 测试
    loader.test(); net.eval();
    int ok = 0, tot = 0;
    while (loader.next_batch(x, y)) {
        auto out = net.forward(x);
        if (out.at(0,0) > out.at(1,0)) { if (y.at(0,0) > y.at(1,0)) ok++; }
        else { if (y.at(1,0) > y.at(0,0)) ok++; }
        tot++;
    }
    printf("Accuracy: %d/%d = %.1f%%\n", ok, tot, 100.0f*ok/tot);
}
```

### LSTM 时序预测

```cpp
// 输入: 100 时间步, 每步 3 特征 → 输出: 100 时间步, 每步 2 特征
auto lstm = new LSTM(/*input=*/3, /*hidden=*/16);
auto linear = new Linear(16, 2);
Sequence seq({lstm, linear});

Matrix<float> x(100, 3);  // 100 步 x 3 特征
Matrix<float> y(100, 2);  // 100 步 x 2 目标
// ... 填充数据 ...
Optim optim(Adam_step, seq.modules(), 0.005f);
for (int e = 0; e < 500; e++) {
    auto out = seq.forward(x);
    optim.step(mse_loss_grad(out, y));
}
```

### CNN 图像分类

```cpp
auto conv1 = new Conv2d(1, 16, 3, 1, 0);   // 28×28×1 → 26×26×16
auto relu1 = new Activation::ReLU();
auto pool1 = new MaxPool2d(2, 2);           // 26×26 → 13×13
auto flat  = new Flatten();                 // 13×13×16 → 2704
auto fc    = new Linear(2704, 10);
auto sm    = new Activation::SoftMax();
Sequence cnn({conv1, relu1, pool1, flat, fc, sm});

// 或用工厂
auto cnn = nn::CNN({1, 16, 32}, 3, nn::ReLU, 10);
```

---

## 10. ONNX 互操作

NPCore 通过 `nn::ONNXModel` 类支持 ONNX 导入/导出，可与 PyTorch、ONNX Runtime 等框架互操作。详见 **[ONNX 支持文档](ONNX.md)**。

```cpp
// 导出: NPCore → ONNX
auto onnx = nn::ONNXModel::from_sequence(model, {1, 28, 28}, "MyModel");
onnx.save("model.onnx");

// 导入: ONNX → NPCore
auto loaded = nn::ONNXModel::load("model.onnx");
Sequence model = loaded.to_sequence();
```

---

## 11. 最佳实践

1. **train() / eval()**: 训练前调用 `layer->train()`, 评估前调用 `layer->eval()` (影响 BatchNorm/Dropout)
2. **梯度裁剪**: RNN 训练时梯度超过 10.0 自动裁剪; 可用 `GradientClipping::clip_by_norm()` 手动控制
3. **参数初始化**: 卷积层默认 `XavierUniform`, 线性层默认 `Uniform`; 可传入 `InitMode::HeNormal` 等
4. **批处理**: `InMemoryLoader(batch_size=32)` 堆叠样本; `SequenceLoader` 自动 padding 变长序列
5. **内存**: 所有层用 `new` 创建, 加入 `Sequence` 后自动管理生命周期 (不要手动 delete)
6. **模型交换**: 用 `save_model`/`load_model_weights` 保存训练 checkpoint；用 `ONNXModel` 与外部框架交换模型
