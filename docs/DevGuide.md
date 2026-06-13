# 开发指南

## 编码规范

### 命名约定

| 类别 | 规范 | 示例 | 说明 |
|------|------|------|------|
| 类名 | PascalCase | `Conv2d`, `BatchNorm1d` | 首字母大写，每个单词大写 |
| 方法名 | camelCase | `forward()`, `getParams()` | 首字母小写，动词开头 |
| 成员变量 | snake_case | `weight_grad_`, `col_cache_` | 下划线分隔，全小写 |
| 私有成员 | 后缀 `_` | `weight_grad_`, `col_cache_` | 只用于 private 成员 |
| 常量 | UPPER 或 PascalCase | `BLOCK_SIZE`, `XavierUniform` | 编译时常量大写 |
| 文件名 | PascalCase | `Conv2d.h`, `CudaBridge.h` | 与主类同名 |
| 命名空间 | PascalCase | `NPCore`, `Activation` | |

### 文件组织原则

```
新层 → 放到对应子目录:
  基础操作 → Layers/Basic/       (Linear, Flatten, Embedding, Dropout)
  卷积     → Layers/Conv/        (Conv2d, MaxPool2d)
  循环     → Layers/Recurrent/   (RNN, LSTM, GRU)
  归一化   → Layers/Normalization/(BatchNorm, LayerNorm)
  架构组件 → Layers/Architecture/ (Residual, ResNetBlock)

新激活函数 → Activations/Activation.h/.cpp  (全部在同一个文件中)
新损失函数 → Losses/Loss.h                   (全部在同一个文件中)
新优化器   → Optimizers/XXX.h               (每个优化器独立文件)
```

### 内存管理铁律

```cpp
// 规则1: 谁创建谁销毁
~MyLayer() override {
    delete weight;    // 构造函数中 new 的
    delete bias;
    delete dW;        // backward() 中 new 的
    delete db;
    // gard/output 中的临时矩阵由 ~Module() 基类自动清理
}

// 规则2: 子层的 gard/output 必须传播清理
void CleanGard() override {
    sublayer1->CleanGard();    // 先清理子层
    sublayer2->CleanGard();
    for (auto p : gard) delete p;    // 再清理自己
    gard.clear();
    for (auto p : output) delete p;
    output.clear();
    delete dW; dW = nullptr;         // 重置梯度
    delete db; db = nullptr;
}

// 规则3: GPU 搬运必须传播
void cuda() override {
    sublayer1->cuda();   // 所有子层都要搬运
    sublayer2->cuda();
    Module::cuda();       // 最后搬运自己的 getParams()
}

// 规则4: 禁止浅拷贝
// Sequence 已 delete 拷贝构造/赋值
// Module 通过虚析构 + 纯虚方法禁止直接实例化
```

---

## 实现原理

### GEMM 矩阵乘法 (参考 GotoBLAS 2008)

```
C(M,N) = A(M,K) × B(K,N)

算法流程:
1. 外层分块 (缓存优化):
   for jc=0..N step NC (384):
     for pc=0..K step KC (256):
       Pack B[pc:pc+KC, jc:jc+NC] → B_packed[KC][NR] (16KB L1)
       for ic=0..M step MC (384):
         for ir=ic..ic+MC step MR (6):
           Pack A[ir:ir+MR, pc:pc+KC] → A_packed[MR][KC] (6KB L1)
           微内核(6×16) → C[ir:ir+6, jr:jr+16]

2. 微内核 (AVX2, 利用全部16个 YMM 寄存器):
   寄存器分配:
     C00..C51: 12 regs - 累加器 (6行 × 2半列 = 12)
     B0, B1:    2 regs - B 面板当前行 (每行16个float, 分2个YMM)
     A0..A5:    逐个广播 - 每轮 k 循环依次广播 A 面板6行的第k个元素
   内循环 (k=0..KC-1):
     B0 = load(B_packed[k][0:8])     // 加载 B 行
     B1 = load(B_packed[k][8:16])
     A_val = broadcast(A_packed[0][k])  // 广播 A 值
     C00 = fmadd(A_val, B0, C00)        // 融合乘加
     C01 = fmadd(A_val, B1, C01)
     // ... 重复6次 (6行 × 2半列 = 12次 FMADD)
   每轮: 2 load + 6 broadcast + 12 fmadd = 理论 24 FLOPS/cycle
   加上预取: _mm_prefetch(B_packed + (k+8)*16) 提前加载下一缓存行

3. Kahan 补偿求和 (标量路径):
   对每个 C[i][j] 累加时:
     y = A[i][k] * B[k][j] - comp    // 减去上次的补偿
     t = sum + y                       // 暂存
     comp = (t - sum) - y              // 计算新的补偿 (损失的精度)
     sum = t                           // 更新和
   相比朴素累加, 将 O(ε√K) 的误差降到 O(ε)
```

### Conv2d: im2col + GEMM

```
输入: (H, W, C_in)   卷积核: (K_h, K_w, C_in, C_out)

Forward:
  1. im2col(input, K_h, K_w, stride, padding):
     将每个 K_h×K_w×C_in 的滑动窗口展平成列
     → col (C_in·K_h·K_w, H_out·W_out) = (patch_size, num_patches)

  2. weight_2d() 将卷积核从 (K_h,K_w,C_in·C_out) 重塑为 (C_out, C_in·K_h·K_w)

  3. result_2d = W_2d × col     → (C_out, H_out·W_out)
     使用上面的 optimized GEMM

  4. reshape + bias → (H_out, W_out, C_out)

Backward:
  dW: G_2d(C_out,HW) × col^T(HW,patch) → (C_out,patch) → reshape
  dB: Σ G over spatial dims
  dX: W_2d^T(patch,C_out) × G_2d(C_out,HW) → col2im → (H,W,C_in)

weight_2d 缓存策略:
  - 首次 forward 时计算并缓存到 weight_2d_cache_
  - getParams() 被调用时设 weight_2d_dirty_ = true
    (优化器通过 getParams()→getAllGrads() 更新权重)
  - 下一次 forward 自动懒重新计算
  - 推理模式下 forward 直接复用缓存 (跳过 getParams 调用, 不会变脏)
```

### RNN 反向传播 (BPTT)

```
前向 (t=0→T-1):
  h_t = tanh(W_ih·x_t + W_hh·h_{t-1} + b_h)
  存储所有 h_t 和 x_t 到缓存 (h_cache, x_cache)

反向 (t=T-1→0):
  dh_total = grad_output[t] + dh_next
            ^^^^^^^^^^^^^^^^   ^^^^^^^^^
            输出层的损失梯度   从未来时间步传来的梯度

  梯度裁剪: if |dh_total| > 10 → dh_total = sign(dh_total) × 10
          防止梯度爆炸 (RNN 常见问题)

  dtanh = dh_total ⊙ (1 - h_t²)           // tanh 导数
  dW_ih += dtanh × x_t^T                   // 输入权重梯度累加
  dW_hh += dtanh × h_{t-1}^T               // 循环权重梯度累加
  db_h  += dtanh                            // 偏置梯度累加
  dx_t = W_ih^T × dtanh                    // 传回输入的梯度
  dh_next = W_hh^T × dtanh                 // 传给 t-1 的梯度

初始化技巧:
  W_hh 初始化后 *= 0.1 (XavierUniform 后缩小, 防止初始发散)
```

### LSTM 门控机制

```
合并权重矩阵 (一次 GEMM 替代 4 次):
  W = concat[W_f, W_i, W_o, W_g]  shape: (4H, I+H)
  输入拼接: xh = [x; h_prev]      shape: (I+H, 1)
  gates = W × xh + b               shape: (4H, 1)

门拆分与应用:
  forget = σ(gates[0:H])           // 遗忘门: 控制旧记忆的保留
  input  = σ(gates[H:2H])          // 输入门: 控制新信息的写入
  output = σ(gates[2H:3H])         // 输出门: 控制记忆的暴露
  cell   = tanh(gates[3H:4H])      // 细胞候选值

  c_t = forget ⊙ c_{t-1} + input ⊙ cell   // 细胞状态更新
  h_t = output ⊙ tanh(c_t)                 // 隐藏状态

梯度流分析:
  dc_{t-1} = dc_t × forget
  当 forget ≈ 1 时, 梯度无衰减跨时间步传播
  当 forget ≈ 0 时, 梯度被阻断 (LSTM 主动"遗忘")

初始化:
  forget bias = +2.0  → σ(2.0) ≈ 0.88 → 默认大部分保留
  input  bias = -2.0  → σ(-2.0)≈ 0.12 → 默认不写入噪声
```

### CUDA 自动分发

```
调用链:
  Matrix::operator*(A,B)
  ├─ if A.device==GPU && B.device==GPU:
  │    cuda_gemm_device(M,N,K, d_A,d_B,d_C)  // GPU 指针, 零拷贝
  │    └─ cudaMalloc + H2D copy + kernel + D2H copy (一次性)
  ├─ else if CudaDevice::available && size>4096:
  │    cuda_gemm_dispatch(M,N,K, A,B,C)      // CPU→GPU, 运算, GPU→CPU
  └─ else:
       gemm(M,N,K, A,B,C)                    // CPU packed GEMM 微内核

模型搬移:
  net.cuda()
  └─ Sequence::cuda()
       └─ for each layer: layer->cuda()
            └─ Module::cuda() → for each W in getParams(): W.to(GPU)
                 └─ Matrix::to(GPU)
                      ├─ cudaMalloc(n*sizeof(float))
                      ├─ cudaMemcpy H2D
                      ├─ delete[] CPU data
                      └─ device_ = GPU
```

---

## 二次开发模板

### 自定义激活函数

```cpp
// 1. 在 Activations/Activation.h 添加类声明
class MyAct : public Module<float> {
public:
    MyAct() = default;
    ~MyAct() override = default;
    Matrix<float> forward(Matrix<float>& x) override;   // 必须
    Matrix<float> backward(Matrix<float>& g) override;  // 必须
    // 以下四个是模板代码, 直接复制
    std::vector<Matrix<float>*> getParams() override { return {}; }
    Matrix<float>* getGard() override { return gard.empty()?nullptr:gard.back(); }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override {
        for(auto p:gard)delete p; gard.clear();
        for(auto p:output)delete p; output.clear();
    }
};

// 2. 在 Activations/Activation.cpp 实现
Matrix<float> MyAct::forward(Matrix<float>& in) {
    // 分配输出矩阵和导数矩阵 (shape 与输入一致)
    auto* r = new Matrix<float>(in.row, in.col, in.channel);
    auto* d = new Matrix<float>(in.row, in.col, in.channel);
    int n = in.row * in.col * in.channel;    // 总元素数

    for (int i = 0; i < n; ++i) {
        float x = in.data_ptr()[i];          // 读取输入
        r->data_ptr()[i] = /* 激活函数 f(x) */;
        d->data_ptr()[i] = /* 导数 f'(x) */;
    }

    gard.push_back(d);    // 存导数 → backward 通过 gard.back() 读取
    output.push_back(r);  // 存输出 → getOutput() 返回
    return *r;
}

Matrix<float> MyAct::backward(Matrix<float>& g) {
    // 标准模式: grad ⊙ derivative
    // g: dL/dy (上游传来的梯度)
    // gard.back(): 存储的 f'(x)
    return gard.empty() ? g : (g & (*gard.back()));  // & 是逐元素乘法
}
```

### 自定义损失函数

```cpp
// 在 Losses/Loss.h 添加两个模板函数

// 1. 损失值 (标量, 用于监控训练)
template<typename T>
T my_loss(const Matrix<T>& pred, const Matrix<T>& target) {
    // pred: 网络输出, target: 目标值, shape 必须一致
    if (pred.row != target.row || pred.col != target.col)
        throw std::runtime_error("my_loss: shape mismatch");

    T total = 0;
    int n = pred.row * pred.col * pred.channel;  // 总元素数
    for (int i = 0; i < n; ++i) {
        T diff = pred.data_ptr()[i] - target.data_ptr()[i];
        total += /* 损失公式, 如 diff*diff */;
    }
    return total / (T)n;  // 返回平均值
}

// 2. 损失梯度 (矩阵, 用于 optim.step)
template<typename T>
Matrix<T> my_loss_grad(const Matrix<T>& pred, const Matrix<T>& target) {
    // 返回 dL/dpred, shape 必须与 pred 一致
    Matrix<T> g(pred.row, pred.col, pred.channel);
    int n = pred.row * pred.col * pred.channel;
    for (int i = 0; i < n; ++i) {
        T diff = pred.data_ptr()[i] - target.data_ptr()[i];
        g.data_ptr()[i] = /* 梯度公式, 如 2*diff */;
    }
    return g;
}

// 使用: 与内置损失完全相同
float loss = my_loss(out, target);
optim.step(my_loss_grad(out, target));
```

### 自定义层 (无子层)

```cpp
class MyLayer : public Module<float> {
    Matrix<float> *W = nullptr;       // 权重
    Matrix<float> *b = nullptr;       // 偏置
    Matrix<float> *dW = nullptr;      // 权重梯度 (backward 时 new)
    Matrix<float> *db = nullptr;      // 偏置梯度

public:
    MyLayer(int in, int out) {
        W = new Matrix<float>(out, in);              // 权重: (out, in)
        b = new Matrix<float>(out, 1);               // 偏置: (out, 1)
        InitMatrixFunc(*W, XavierUniform);           // Xavier 初始化权重
        InitMatrixFunc(*b, Zeros);                   // 偏置初始化为0
    }
    ~MyLayer() override { delete W; delete b; delete dW; delete db; }

    // 1. 前向传播
    Matrix<float> forward(Matrix<float>& x) override {
        if (train_mode)
            gard.push_back(new Matrix<float>(x));    // 存输入给 backward

        auto* out = new Matrix<float>((*W) * x + (*b));
        output.push_back(out);
        return *out;
    }

    // 2. 反向传播
    Matrix<float> backward(Matrix<float>& g) override {
        auto& x = *gard.back();                      // 读取存储的输入
        dW = new Matrix<float>(g * x.Translate());   // dW = g × x^T
        db = new Matrix<float>(g);                   // db = g
        return W->Translate() * g;                   // dx = W^T × g
    }

    // 3-7: 标准模板方法
    std::vector<Matrix<float>*> getParams() override { return {W, b}; }
    std::vector<Matrix<float>*> getAllGrads() override { return {dW, db}; }
    Matrix<float>* getGard() override { return gard.empty()?nullptr:gard.back(); }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void CleanGard() override {
        for (auto p : gard) delete p; gard.clear();
        for (auto p : output) delete p; output.clear();
        delete dW; dW = nullptr;
        delete db; db = nullptr;
    }
};
```

### 自定义层 (含子层)

```cpp
// 当你的层内部包含其他 Module (如 Conv2d, ReLU):
class MyBlock : public Module<float> {
    Conv2d *c1, *c2;             // 子层 (你拥有所有权)
    Activation::ReLU *r1, *r2;

public:
    MyBlock(int ch) {
        c1 = new Conv2d(ch, ch, 3, 1, 1);
        c2 = new Conv2d(ch, ch, 3, 1, 1);
        r1 = new Activation::ReLU();
        r2 = new Activation::ReLU();
    }
    ~MyBlock() override { delete c1; delete c2; delete r1; delete r2; }

    Matrix<float> forward(Matrix<float>& x) override {
        auto o = r2->forward(c2->forward(r1->forward(c1->forward(x))));
        auto* s = new Matrix<float>(o);
        output.push_back(s);
        return *s;
    }
    Matrix<float> backward(Matrix<float>& g) override {
        g = r2->backward(g);
        g = c2->backward(g);
        g = r1->backward(g);
        g = c1->backward(g);
        return g;
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

    // ★ 关键: 必须传播到子层 ★
    void cuda() override { c1->cuda(); c2->cuda(); }
    void cpu()  override { c1->cpu();  c2->cpu();  }
    void eval()  { train_mode=false; c1->eval();  c2->eval();  }
    void train() { train_mode=true;  c1->train(); c2->train(); }

    void CleanGard() override {
        c1->CleanGard(); c2->CleanGard();    // 必须先清理子层!
        r1->CleanGard(); r2->CleanGard();
        for (auto p : gard) delete p; gard.clear();
        for (auto p : output) delete p; output.clear();
    }

    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
};
```

### 添加到项目

```
1. 文件放到对应子目录
2. 在 cmake/NPCoreSources.cmake 中注册
3. 在 NPCore/NN.h (或对应模块头) 添加 #include

完成。之后 cmake --build _build 自动编译。
```

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| backward 中 gard.back() 为空 | eval 模式跳过了存储 | 调用 `net.train()` |
| 3D 矩阵逐元素运算报错 | operator+/& 旧版不支持 channel | 已修复, 确保使用最新 Matrix.inl |
| Sequence 中自定义层报 shape mismatch | forward 返回值 shape 与下一层不匹配 | 检查 forward 输出的 row/col/channel |
| 权重不更新 | CleanGard 删除了梯度但 getAllGrads 返回 nullptr | 确保 backward 中 new 了梯度, CleanGard 在 optim.step 之后才调用 |
| 含子层的块 GPU 搬运无效 | 没有重写 cuda() 传播到子层 | 见上方"自定义层(含子层)"模板 |
| CUDA 编译找不到 cl.exe | MSVC Build Tools 未安装 | 安装 VS2022 Build Tools, 勾选 C++ 工作负载 |
