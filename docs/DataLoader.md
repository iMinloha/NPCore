# DataLoader 数据加载指南

NPCore 提供完整的数据加载框架，支持表格数据、图像、可变长序列等多种数据格式。

## 目录

- [快速开始](#快速开始)
- [内置 Loader](#内置-loader)
  - [InMemoryLoader — 内存数据](#inmemoryloader)
  - [CSVLoader — 表格数据](#csvloader)
  - [ImageFolderLoader — 图像数据](#imagefolderloader)
  - [SequenceLoader — 变长序列](#sequenceloader)
  - [BatchStackLoader — 批处理包装](#batchstackloader)
- [自定义 Loader](#自定义-loader)
- [与 Trainer 集成](#与-trainer-集成)
- [数据格式约定](#数据格式约定)

---

## 快速开始

最简单的用法 — 使用内置 InMemoryLoader:

```cpp
#include "NPCore.h"
using namespace NPCore;

int main() {
    // 1. 准备数据
    InMemoryLoader loader(/*batch_size=*/1);
    for (int i = 0; i < 100; ++i) {
        Matrix<float> x(4, 1);  // 4 个特征
        Matrix<float> y(1, 1);  // 1 个目标
        x << (float)i << (float)(i*2) << (float)(i%10) << (float)(i/10);
        y << (float)(i % 2);
        loader.add_sample(x, y);
    }

    // 2. 划分训练集/测试集
    loader.split(0.8f, 0.2f);  // 80% train, 20% test

    // 3. 训练
    auto net = nn::FNN({4, 8, 4, 1}, nn::Sigmoid);
    loader.train();  // 切换到训练集
    nn::Trainer trainer(net, nn::MSE, Optim(net.getParams(), Adam, 0.01f));
    trainer.fit(loader, 100, [](int e, float loss) {
        printf("epoch %d: %.6f\n", e, loss);
    });
}
```

---

## 内置 Loader

### InMemoryLoader

从 `std::vector<Matrix<float>>` 加载，支持 train/test/val 划分。

```cpp
// 创建
InMemoryLoader loader(int batch_size = 1, bool shuffle = true);

// 添加样本
loader.add_sample(input_matrix, target_matrix);

// 划分数据集
loader.split(0.7f, 0.2f, 0.1f);  // 70% train, 20% test, 10% val

// 切换 split
loader.train();   // 训练模式
loader.test();    // 测试模式
loader.val();     // 验证模式

// 迭代
Matrix<float> x, y;
while (loader.next_batch(x, y)) {
    // x, y 是当前样本
}

// batch_size > 1 时自动按行堆叠
InMemoryLoader batched(32);  // 每次返回 32 个样本堆叠的矩阵
// x shape: (32, features), y shape: (32, targets)
```

**数据格式:**

| 网络类型 | 输入 x shape | 目标 y shape |
|---------|------------|------------|
| FNN (Linear) | `(features, 1)` 列向量 | `(classes, 1)` |
| CNN (Conv2d) | `(H, W, C)` 3D tensor | `(classes, 1)` |
| RNN/LSTM/GRU | `(seq_len, input_size)` | `(seq_len, hidden_size)` |

### CSVLoader

从 CSV 文件或字符串加载表格数据。

```cpp
// 从文件加载 — 最后一列是目标值
CSVLoader csv("housing.csv", /*target_cols=*/1, /*has_header=*/true);
csv.load();  // 解析文件
csv.split(0.8f, 0.2f);

// 从字符串加载
const char* data = R"(
    1.0, 2.0, 0.5
    3.0, 4.0, 0.8
    5.0, 6.0, 1.2
)";
CSVLoader csv2("", 1);
csv2.parse(data);

// 手动添加行
csv.add_row({1.0f, 2.0f, 3.0f}, {0.5f});  // features={1,2,3}, target={0.5}

// 多目标列
CSVLoader multi("multi.csv", /*target_cols=*/3);  // 前 3 列是目标，其余是特征
```

**CSV 格式:** 每行一个样本，前 `n_target_cols` 列是目标值，后续列是特征。支持双引号包裹字段。

### ImageFolderLoader

从文件夹加载图像，按子目录分类。

```
data/images/
  cat/      img001.png  img002.png ...
  dog/      img001.png  img002.png ...
  bird/     img001.png  img002.png ...
```

```cpp
#include <stb_image.h>  // 或任何图像库

// 1. 定义图像解码函数
auto decode = [](const std::string& path) -> Matrix<float> {
    int w, h, c;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 3);
    Matrix<float> img(h, w, 3);
    int idx = 0;
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            for (int ch = 0; ch < 3; ++ch)
                img.at(i, j, ch) = data[idx++] / 255.0f;
    stbi_image_free(data);
    return img;
};

// 2. 扫描目录
ImageFolderLoader dataset("data/images", decode, /*batch_size=*/32);
int n = dataset.scan();  // 返回发现的图像数量
printf("Found %d images in %d classes\n", n, dataset.num_classes());

// 3. 查看类别
for (auto& name : dataset.classes())
    printf("  class: %s\n", name.c_str());

// 4. 划分并训练
dataset.split(0.8f, 0.2f);
dataset.train();

Matrix<float> x, y;
while (dataset.next_batch(x, y)) {
    // x: (H, W, C) 3D tensor — 单张图像
    // y: (num_classes, 1) — one-hot 标签
}
```

**支持格式:** `.png` `.jpg` `.jpeg` `.bmp` `.tga` `.ppm` (通过扩展名筛选)

### SequenceLoader

处理可变长度序列，自动 padding 到批次最大长度。

```cpp
SequenceLoader seq_loader(/*batch_size=*/16);

// 添加不等长序列
Matrix<float> s1(5, 3);   // 长度 5, 3 个特征
Matrix<float> t1(5, 1);   // 长度 5, 1 个目标
// ... 填充数据 ...
seq_loader.add_sequence(s1, t1);

Matrix<float> s2(8, 3);   // 长度 8 — 不同长度
Matrix<float> t2(8, 1);
seq_loader.add_sequence(s2, t2);

seq_loader.split(0.8f, 0.2f);
seq_loader.train();

// 获取 padded batch + mask
Matrix<float> x, y, mask;
while (seq_loader.next_batch(x, y, mask)) {
    // x: (max_len_in_batch, features * batch_size)
    // y: (max_len_in_batch, targets * batch_size)
    // mask: (max_len_in_batch, batch_size) — 1.0=有效, 0.0=padding

    // 使用 mask 计算损失时忽略 padding 位置
}
```

### BatchStackLoader

将任意 DataLoader 包装为批处理模式。

```cpp
InMemoryLoader raw;
// ... 添加样本 ...
BatchStackLoader batched(&raw, /*batch_size=*/32);

// 普通批处理 (FNN/CNN) — 按行堆叠
Matrix<float> x, y;
batched.next_batch(x, y);  // x: (32, features)

// RNN 批处理 — padding 到 batch 内最大长度
BatchStackLoader rnn_batch(&raw, 32, /*pad_sequences=*/true);
batched.next_batch(x, y);
// 获取原始长度 (用于 unpack)
auto& lengths = batched.last_lengths();
```

---

## 自定义 Loader

继承 `DataLoader` 实现 `num_samples()` 和 `next_batch()` 即可:

```cpp
class MyDatabaseLoader : public DataLoader {
    DatabaseConnection db_;
    std::string current_table_;
    int cursor_ = 0;
    int total_ = 0;

public:
    MyDatabaseLoader(const std::string& conn_str) : db_(conn_str) {}

    // 必须实现: 当前 split 的样本数
    int num_samples() const override { return total_; }

    // 必须实现: 获取下一批数据
    bool next_batch(Matrix<float>& x, Matrix<float>& y) override {
        if (cursor_ >= total_) { cursor_ = 0; return false; }

        auto row = db_.fetch_row(current_table_, cursor_);
        x = Matrix<float>(row.features.size(), 1);
        y = Matrix<float>(row.targets.size(), 1);
        // ... 填充矩阵 ...
        cursor_++;
        return true;
    }

    // 可选: 切换数据集
    void set_split(DataSplit s) override {
        current_table_ = (s == DataSplit::Train) ? "train_data"
                       : (s == DataSplit::Test)  ? "test_data"
                       : "val_data";
        total_ = db_.count_rows(current_table_);
        cursor_ = 0;
    }

    // 可选: 重置迭代器
    void reset() override { cursor_ = 0; }
};
```

### 必须实现的接口

| 方法 | 说明 |
|------|------|
| `int num_samples() const` | 返回当前 split 的样本总数 |
| `bool next_batch(x, y)` | 填充输入/目标矩阵，返回 false 表示 epoch 结束 |

### 可选重写

| 方法 | 说明 |
|------|------|
| `void reset()` | epoch 开始时的重置操作 |
| `void set_split(DataSplit)` | 处理 train/test/val 切换 |

---

## 与 Trainer 集成

```cpp
// 方式 1: Trainer 直接接受 DataLoader
trainer.fit(loader, epochs, callback);

// 方式 2: 手动训练循环
loader.train();
for (int epoch = 0; epoch < 100; ++epoch) {
    Matrix<float> x, y;
    float epoch_loss = 0;
    int batches = 0;
    while (loader.next_batch(x, y)) {
        auto out = net.forward(x);
        float loss = compute_loss(out, y);
        net.backward(grad_of_loss);
        optimizer.step();
        epoch_loss += loss;
        batches++;
    }
    printf("epoch %d: avg loss %.6f\n", epoch, epoch_loss / batches);
}
```

---

## 数据格式约定

| 网络类型 | x 形状 | y 形状 | 说明 |
|---------|--------|--------|------|
| Linear / FNN | `(features, 1)` | `(classes, 1)` | 列向量 |
| Conv2d | `(H, W, C)` | `(classes, 1)` | 3D tensor |
| RNN / LSTM / GRU | `(seq_len, input_size)` | `(seq_len, hidden_size)` | 行=时间步 |
| MultiHeadAttention | `(seq_len, d_model)` | `(seq_len, d_model)` | 自注意力 |

**批处理规则:**
- FNN: `(features, 1)` x N -> `(N, features)` 按行堆叠
- CNN: 保持 `(H, W, C)` 独立返回 (由 Trainer 循环处理)
- RNN: padding 到批次最大长度，附带 mask 矩阵
