# 开发自己的模型

## 5 分钟快速开发

继承 `Module<float>`，实现 `forward()` 和 `backward()`：

```cpp
#include "CorePP.h"
using namespace CoreNNSpace;

// 1. 定义你的模型
class MyLayer : public Module<float> {
    Matrix<float> *W, *b, *dW = nullptr, *db = nullptr;
    int in_dim, out_dim;
public:
    MyLayer(int in_dim, int out_dim) : in_dim(in_dim), out_dim(out_dim) {
        W = new Matrix<float>(out_dim, in_dim);
        b = new Matrix<float>(out_dim, 1);
        InitMatrixFunc(*W, XavierUniform);
        InitMatrixFunc(*b, Zeros);
    }
    ~MyLayer() override { delete W; delete b; delete dW; delete db; }

    // 2. 前向传播: y = W*x + b
    Matrix<float> forward(Matrix<float>& x) override {
        if (train_mode) gard.push_back(new Matrix<float>(x)); // 存输入给 backward
        auto* out = new Matrix<float>((*W) * x + (*b));
        output.push_back(out);
        return *out;
    }

    // 3. 反向传播
    Matrix<float> backward(Matrix<float>& grad) override {
        auto& x = *gard.back();
        dW = new Matrix<float>(grad * x.Translate());  // dW = grad * x^T
        db = new Matrix<float>(grad);                   // db = grad
        return W->Translate() * grad;                   // dx = W^T * grad
    }

    // 4. 参数接口
    std::vector<Matrix<float>*> getParams() override { return {W, b}; }
    std::vector<Matrix<float>*> getAllGrads() override { return {dW, db}; }
    Matrix<float>* getGard() override { return gard.empty()?nullptr:gard.back(); }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }

    void CleanGard() override {
        for (auto p : gard) delete p; gard.clear();
        for (auto p : output) delete p; output.clear();
        delete dW; dW = nullptr; delete db; db = nullptr;
    }
};

// 5. 使用
int main() {
    MyLayer layer(4, 8);
    Sequence net({&layer, new Activation::ReLU(), new Linear(8, 4)});

    Matrix<float> x(4,1); x << 1 << 2 << 3 << 4;
    Matrix<float> y(4,1); y << 1 << 0 << 0 << 0;

    Optim optim(net.getParams(), Adam, 0.01f);
    for (int e = 0; e < 100; e++) {
        auto out = net.forward(x);
        optim.step(out - y);
    }
}
```

## 关键规则

| 规则 | 说明 |
|------|------|
| `forward()` | 必须在 `gard` 中存储需要用于 `backward` 的中间值 |
| `backward()` | 必须设置 `dW`/`db`(或 `getAllGrads()`) 供优化器读取 |
| `getParams()` | 返回所有权重矩阵指针，顺序与 `getAllGrads()` 对应 |
| `CleanGard()` | 清理所有梯度和缓存，每次 optimizer step 后调用 |
| `train_mode` | 为 `true` 时存储梯度缓存，`eval()` 时跳过 |

## 自动获得的能力

只要实现了上述接口，你的模型**自动**获得：

- **GPU 加速** - `net.cuda()` 一行搬运所有参数
- **所有优化器** - SGD/Adam/RMSProp/AdamW
- **梯度检验** - `numerical_gradient()` + `gradcheck()`
- **Sequence 组合** - 与其他层自由组合
- **Trainer API** - `nn::Trainer(net, ...).fit(...)`

## 进阶: 组合多个子层

如果你的模型内部包含其他 Module（如 ResNet 的 Conv+ReLU+Conv），需要重写更多方法：

```cpp
class MyResBlock : public Module<float> {
    Conv2d *c1, *c2;
    Activation::ReLU *r1, *r2;
    Matrix<float>* skip_cache = nullptr;

public:
    MyResBlock(int ch) {
        c1 = new Conv2d(ch, ch, 3, 1, 1);
        c2 = new Conv2d(ch, ch, 3, 1, 1);
        r1 = new Activation::ReLU();
        r2 = new Activation::ReLU();
    }
    ~MyResBlock() override { delete c1; delete c2; delete r1; delete r2; delete skip_cache; }

    // forward: 两个 Conv + skip connection
    Matrix<float> forward(Matrix<float>& x) override {
        delete skip_cache; skip_cache = new Matrix<float>(x);  // 保存输入
        auto o = c1->forward(x);
        o = r1->forward(o);
        o = c2->forward(o);
        // skip: element-wise add (manual, preserves 3D shape)
        int n = o.row * o.col * o.channel;
        for (int i = 0; i < n; ++i) o.data_ptr()[i] += skip_cache->data_ptr()[i];
        o = r2->forward(o);
        auto* s = new Matrix<float>(o);
        output.push_back(s);
        return *s;
    }

    // backward: 梯度通过两条路径
    Matrix<float> backward(Matrix<float>& g) override {
        auto g0 = r2->backward(g);
        auto gs = Matrix<float>(g0);        // skip 路径梯度
        g0 = c2->backward(g0);              // main 路径
        g0 = r1->backward(g0);
        g0 = c1->backward(g0);
        int n = g0.row * g0.col * g0.channel;
        for (int i = 0; i < n; ++i) g0.data_ptr()[i] += gs.data_ptr()[i];
        return g0;
    }

    // 聚合子层参数
    std::vector<Matrix<float>*> getParams() override {
        auto p = c1->getParams();
        for (auto* m : c2->getParams()) p.push_back(m);
        return p;
    }
    std::vector<Matrix<float>*> getAllGrads() override {
        auto g = c1->getAllGrads();
        for (auto* m : c2->getAllGrads()) g.push_back(m);
        return g;
    }

    // GPU / 训练模式 传播到子层
    void cuda() override { c1->cuda(); c2->cuda(); }
    void cpu()  override { c1->cpu();  c2->cpu();  }
    void eval()  { train_mode=false; c1->eval();  c2->eval();  }
    void train() { train_mode=true;  c1->train(); c2->train(); }

    // 清理所有子层
    void CleanGard() override {
        for (auto p : gard) delete p;     gard.clear();
        for (auto p : output) delete p;    output.clear();
        c1->CleanGard(); c2->CleanGard();
        r1->CleanGard(); r2->CleanGard();
    }

    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
};
```

## 必须重写的方法

| 方法 | 简单层 (无子层) | 复合层 (有子层) |
|------|----------------|-----------------|
| `forward()` | 自己实现 | 调用子层 forward |
| `backward()` | 自己实现 | 调用子层 backward |
| `getParams()` | `{W, b}` | 聚合所有子层 |
| `getAllGrads()` | `{dW, db}` | 聚合所有子层梯度 |
| `CleanGard()` | 清自己 | **必须清所有子层** |
| `cuda()/cpu()` | 默认 | **必须传播到子层** |
| `eval()/train()` | 默认 | **必须传播到子层** |

## getParams / getAllGrads 多参数原理

`getParams()` 和 `getAllGrads()` 返回的是 `vector`，支持**任意数量**的参数。RNN 有 3 个参数就是个例子：

```cpp
// RNN: W_ih(输入权重), W_hh(循环权重), b_h(偏置) - 3 个独立的参数矩阵
std::vector<Matrix<float>*> getParams() override {
    return {W_ih, W_hh, b_h};           // 3 个权重
}
std::vector<Matrix<float>*> getAllGrads() override {
    return {dW_ih, dW_hh, db_h};        // 3 个梯度, 顺序必须一致
}
```

优化器内部用循环逐对处理，不管你有几个参数：

```cpp
// Optimizer.cpp 中的通用更新逻辑:
auto param_list = layer->getParams();   // [W1, W2, W3, ...]  任意长度
auto grad_list  = layer->getAllGrads(); // [dW1, dW2, dW3, ...] 一一对应

for (size_t p = 0; p < param_list.size() && p < grad_list.size(); ++p) {
    if (grad_list[p] != nullptr)
        *param_list[p] = *param_list[p] - (*grad_list[p]) * lr;
}
```

**唯一要求**: `getParams()[i]` 和 `getAllGrads()[i]` 必须指向**同一个参数和它的梯度**。顺序任意，只要一一对应。

梯度可以为 `nullptr` (无参数层如 ReLU 返回空 vector，优化器会跳过)。

```cpp
// 无参数层 (激活函数): 返回空
std::vector<Matrix<float>*> getParams() override { return {}; }

// 单参数层 (Embedding): 1 对
return {weight};       // getParams
return {dW};           // getAllGrads

// 双参数层 (Linear, Conv2d): 2 对
return {weight, bias};
return {weight_grad_, bias_grad_};

// 三参数层 (RNN): 3 对
return {W_ih, W_hh, b_h};
return {dW_ih, dW_hh, db_h};

// 复合层 (ResNetBlock): 聚合子层 → 4 对 (conv1.W, conv1.b, conv2.W, conv2.b)
auto p = c1->getParams();              // [W1, b1]
for (auto* m : c2->getParams())        // [W2, b2]
    p.push_back(m);
return p;                              // → [W1, b1, W2, b2]
```

不需要额外配置 - 只要你的 `getParams()` 和 `getAllGrads()` 返回的 vector 长度一致、顺序对应，优化器自动处理一切。
