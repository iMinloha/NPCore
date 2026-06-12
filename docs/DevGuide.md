# 开发指南

## 编码规范

### 命名约定

| 类别 | 规范 | 示例 |
|------|------|------|
| 类名 | PascalCase | `Conv2d`, `BatchNorm1d` |
| 方法 | camelCase | `forward()`, `getParams()` |
| 成员变量 | snake_case | `weight_grad_`, `train_mode` |
| 私有成员 | 后缀 `_` | `col_cache_`, `learn_rate` |
| 常量 | PascalCase 或 UPPER | `XavierUniform`, `BLOCK_SIZE` |
| 文件 | PascalCase | `Conv2d.h`, `CudaBridge.h` |

### 文件组织

- 每个 `.h` 对应一个主要类，复杂的配 `.cpp`
- 简单层 (Dropout, Pool) 可以纯头文件实现
- 模板代码放在 `.inl` 或 `.hpp` 中
- 相关类型归入子目录 (`Basic/`, `Conv/`, `Recurrent/`)

### 内存管理

```cpp
// 所有权规则:
// 1. Module 拥有其参数矩阵 (new/delete)
// 2. Sequence 拥有其子层 (new/delete)
// 3. gard/output 中的临时矩阵在 CleanGard() 中释放
// 4. 不允许浅拷贝 — 拷贝构造/赋值应 delete 或深拷贝

// 正确的析构:
~MyLayer() override {
    delete weight; delete bias;     // 自己创建的
    delete dW; delete db;           // backward 创建的梯度
    // gard/output 由 ~Module() 自动清理
}
```

## 核心实现原理

### Matrix 乘法 (GEMM)

```
输入: A(M,K) × B(K,N) → C(M,N)
算法: GotoBLAS 风格 packed GEMM

1. 分块: 外层 MC×NC×KC 遍历
2. Pack A: 6×Kc 列优先面板 → L1 缓存
3. Pack B: Kc×16 行优先面板 → L1 缓存
4. 微内核: 6×16 AVX2, 12个YMM存C, 2个存B, 1个广播A
5. 每轮 k 循环: 1×load A + 2×load B → 12×FMADD

关键宏:
- AlgorithmAccelerator: 启用分块 (默认开启)
- __AVX__: CMake 自动检测，启用 AVX2 路径
- 阈值 < 4096 FLOPs: 跳过 packing，直接用标量循环
- Kahan 补偿求和: 标量路径减少浮点累积误差
```

### Conv2d 卷积

```
Forward:  im2col → GEMM → reshape
  im2col: (H,W,C) → (C·K², H_out·W_out)
  weight_2d: (K,K,C_in·C_out) → (C_out, C_in·K²)
  result = W_2d × col → (C_out, H_out·W_out)
  reshape + bias → (H_out, W_out, C_out)

Backward:
  dW = G_2d × col^T → reshape → (K,K,C_in·C_out)
  dB = sum G over spatial dims
  dX = col2im(W_2d^T × G_2d)

缓存: weight_2d 懒更新
  - forward 首次调用时计算并缓存
  - getParams() 访问时置脏标记 (优化器更新权重后)
  - 下次 forward 自动重新计算
```

### RNN 反向传播 (BPTT)

```
前向: h_t = tanh(W_ih·x_t + W_hh·h_{t-1} + b_h)

反向 (t = T-1 → 0):
  dh_total = grad_output[t] + dh_next          // 输出梯度 + 未来梯度
  dtanh = dh_total ⊙ (1 - h_t²)               // tanh 导数
  dW_ih += dtanh × x_t^T                        // 输入权重梯度
  dW_hh += dtanh × h_{t-1}^T                    // 循环权重梯度
  db_h  += dtanh                                // 偏置梯度
  dx_t = W_ih^T × dtanh                         // 输入梯度
  dh_next = W_hh^T × dtanh                      // 向 t-1 传播

梯度裁剪: |dh_total| > 10 → clip to ±10
W_hh 缩放: 初始化后 ×0.1 (防止梯度爆炸)
```

### LSTM 门控机制

```
合并 GEMM: gates(4H) = W(4H, I+H) × [x; h] + b(4H)

门拆分:
  forget = σ(gates[0:H])          // 遗忘门: 丢弃旧记忆
  input  = σ(gates[H:2H])         // 输入门: 写入新信息
  output = σ(gates[2H:3H])        // 输出门: 暴露记忆
  cell   = tanh(gates[3H:4H])     // 细胞门: 候选值

状态更新:
  c_t = forget ⊙ c_{t-1} + input ⊙ cell
  h_t = output ⊙ tanh(c_t)

初始化技巧:
  forget bias = +2  → 默认记住 (σ(2)≈0.88)
  input bias  = -2  → 默认不写 (σ(-2)≈0.12)
  循环权重 ×0.1   → 稳定训练

梯度流:
  dc/dc_{t-1} = forget → 梯度可跨时间步流动
  当 forget≈1 时，梯度无衰减
```

### CUDA 自动分发

```
Matrix::operator*:
  if (双方都是 GPU) → cuda_gemm_device (GPU 指针, 无拷贝)
  else if (CUDA 可用 && 矩阵够大) → cuda_gemm_dispatch (H2D→kernel→D2H)
  else → CPU GEMM 微内核

Activation::forward:
  if (CUDA 可用 && 元素够多) → cuda_sigmoid/tanh (GPU 原地)
  else → CPU 循环

模型搬移:
  net.cuda() → 递归调用所有子层的 cuda()
  → 每个 Matrix::to(Device::GPU)
  → cudaMalloc + cudaMemcpy H2D
```

## 二次开发

### 自定义激活函数

```cpp
// 1. 在 Activations/Activation.h 添加声明
class MyAct : public Module<float> {
public:
    MyAct() = default;
    ~MyAct() override = default;
    Matrix<float> forward(Matrix<float>&) override;
    Matrix<float> backward(Matrix<float>&) override;
    // 这四个方法是模板，复制其他激活函数即可:
    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return gard.empty()?nullptr:gard.back(); }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override { for(auto p:gard)delete p;gard.clear(); for(auto p:output)delete p;output.clear(); }
};

// 2. 在 Activation.cpp 实现
Matrix<float> MyAct::forward(Matrix<float>& in) {
    auto* r = new Matrix<float>(in.row, in.col, in.channel);
    auto* d = new Matrix<float>(in.row, in.col, in.channel);
    int n = in.row * in.col * in.channel;
    for (int i = 0; i < n; ++i) {
        float x = in.data_ptr()[i];
        r->data_ptr()[i] = /* 你的激活函数 */;
        d->data_ptr()[i] = /* 导数 */;
    }
    gard.push_back(d);        // 存导数给 backward
    output.push_back(r);      // 存输出
    return *r;
}

Matrix<float> MyAct::backward(Matrix<float>& g) {
    return gard.empty() ? g : (g & (*gard.back()));  // grad ⊙ derivative
}
```

### 自定义损失函数

```cpp
// 在 Losses/Loss.h 添加 (模板函数, 头文件即可)

template<typename T>
T my_loss(const Matrix<T>& pred, const Matrix<T>& target) {
    if (pred.row != target.row || pred.col != target.col)
        throw std::runtime_error("shape mismatch");
    T total = 0;
    int n = pred.row * pred.col * pred.channel;
    for (int i = 0; i < n; ++i) {
        T d = pred.data_ptr()[i] - target.data_ptr()[i];
        total += /* 你的损失公式 */;
    }
    return total / (T)n;
}

template<typename T>
Matrix<T> my_loss_grad(const Matrix<T>& pred, const Matrix<T>& target) {
    // 返回 dL/dpred, shape 与 pred 相同
    Matrix<T> g(pred.row, pred.col, pred.channel);
    int n = pred.row * pred.col * pred.channel;
    for (int i = 0; i < n; ++i) {
        g.data_ptr()[i] = /* 你的梯度公式 */;
    }
    return g;
}
```

### 自定义层 (无子层)

继承 `Module<float>`, 实现 6 个纯虚方法:

```cpp
class MyLayer : public Module<float> {
    Matrix<float> *W, *b, *dW=nullptr, *db=nullptr;

    // 必须实现:
    Matrix<float> forward(Matrix<float>& x) override;     // 前向 + 存缓存
    Matrix<float> backward(Matrix<float>& g) override;    // 反向 + 设梯度
    std::vector<Matrix<float>*> getParams() override;     // 权重列表
    std::vector<Matrix<float>*> getAllGrads() override;   // 梯度列表
    Matrix<float>* getGard() override;                    // 最新缓存
    Matrix<float>* getOutput() override;                  // 最新输出
    void CleanGard() override;                            // 清理梯度/缓存

    // 可选重写:
    void cuda() override;    // GPU (默认: 搬运 getParams())
    void cpu() override;     // CPU (默认: 搬运 getParams())
};
```

### 自定义层 (含子层)

子层指你的层内部有 `Conv2d*`, `ReLU*` 等其他 Module:

```cpp
class MyBlock : public Module<float> {
    Conv2d *c1, *c2;
    // 除上述 7 个方法外, 还必须重写:
    void cuda() override { c1->cuda(); c2->cuda(); }  // 传播到子层!
    void cpu()  override { c1->cpu();  c2->cpu();  }
    void eval()  { train_mode=false; c1->eval();  c2->eval();  }
    void train() { train_mode=true;  c1->train(); c2->train(); }
    void CleanGard() override {
        // 必须清理子层!
        c1->CleanGard(); c2->CleanGard();
        for(auto p:gard)delete p; gard.clear();
        for(auto p:output)delete p; output.clear();
    }
};
```

### 添加到构建系统

1. 文件放到对应子目录 (`Layers/Basic/`, `Activations/`, 等)
2. 在 `cmake/CorePPSources.cmake` 中添加:
   ```cmake
   list(APPEND COREPP_LAYERS_SOURCES CorePP/Layers/Basic/MyLayer.cpp)
   list(APPEND COREPP_LAYERS_HEADERS CorePP/Layers/Basic/MyLayer.h)
   ```
3. 在 `CorePP/NN.h` (或对应模块头) 添加 `#include`

### GPU 兼容性

你的层**自动**获得 GPU 能力，只要:
1. 所有运算通过 `Matrix::operator*` / `operator+` 等 → 自动检测 CUDA
2. 使用 `Activation::` 激活函数 → 自动检测 CUDA
3. 调用 `net.cuda()` → 递归搬运所有权重

不需要写任何 CUDA 代码。

### 常见问题

**Q: 为什么 backward 中 `gard.back()` 是空的？**
A: forward 时 `train_mode==false` (推理模式) 跳过了存储。调用 `net.train()` 确保训练模式。

**Q: 3D 矩阵 (图像) 的逐元素运算有问题？**
A: 确保使用 `Matrix(row,col,channel)` 构造，`operator+`/`-`/`*` 已支持 3D。

**Q: 自定义层在 Sequence 中报 shape mismatch？**
A: 检查 forward 返回值 shape 是否与下一层期望匹配。Sequence 将每层输出作为下一层输入。

**Q: 如何用 CUDA？**
A: `net.cuda()` 后所有权重和中间结果自动在 GPU 上。需要查看结果时 `out.cpu()`。
