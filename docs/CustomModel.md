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

- **GPU 加速** — `net.cuda()` 一行搬运所有参数
- **所有优化器** — SGD/Adam/RMSProp/AdamW
- **梯度检验** — `numerical_gradient()` + `gradcheck()`
- **Sequence 组合** — 与其他层自由组合
- **Trainer API** — `nn::Trainer(net, ...).fit(...)`

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
