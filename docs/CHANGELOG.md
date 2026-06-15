# Changelog — 2026-06-15 修复与新增

## Bug 修复

### 1. `nn::Tanh()` 返回的是 Sigmoid 而不是 Tanh
- **文件**: `NPCore/Model.cpp:9`
- **问题**: `Module<float>* Tanh() { return new Activation::Sigmoid(); }`
- **修复**: → `return new Activation::Tanh();`

### 2. Trainer 训练不更新任何参数
- **文件**: `NPCore/Model.h`, `NPCore/Model.cpp`, `NPCore/Optimizers/Optimizer.h`, `NPCore/Optimizers/Optimizer.cpp`
- **问题**: `Optim` 用空 params `{}` 构造，`Trainer` 从不从模型提取层参数注入优化器，导致 `step()` 内部 `for` 循环直接跳过，所有权重保持不变。
- **修复**:
  - `Trainer` 改为接受 `Sequence&`（而非 `Module<float>&`），构造时自动调用 `model.getParams()` 提取层列表
  - `Optim` 新增 `set_params()` 方法，`Trainer::bind()` 也会重新绑定参数
  - `Module::eval()/train()` 改为 `virtual`，`Residual`/`ResNetBlock` 加 `override`

### 3. `NPCore::SGD/Adam/RMSProp` 等 8 个 factory 函数声明但未实现
- **文件**: `NPCore/Optimizers/Optimizer.h`, `NPCore/Optimizers/Optimizer.cpp`
- **问题**: `Optimizer.h` 声明了 `SGD()/Adam()/RMSProp()` 但没有实现（链接错误）
- **修复**: 补全 8 个 factory: `SGD/Momentum/Adam/RMSProp/Adagrad/Adadelta/NAdam/RAdam`

### 4. `Matrix::operator+(float)` 和 `operator/(float)` 忽略 channel 维度
- **文件**: `NPCore/Core/Matrix.cpp:46,72`
- **问题**: 这两个运算符只用 `row * col` 计算元素总数，channel > 1 时访问越界
- **修复**: → `row * col * channel`

### 5. `Residual`/`ResNetBlock` 的 `eval()/train()` 不是 override
- **文件**: `NPCore/Layers/Module.h`, `NPCore/Layers/Architecture/Residual.h`, `NPCore/Layers/Architecture/ResNetBlock.h`
- **问题**: `Module::eval()/train()` 不是 virtual，子类的同名函数隐藏了基类版本→ 通过 `Module<float>*` 指针调用时无法触发子类行为（BatchNorm running stats 不切换等）
- **修复**: 基类加 `virtual`，子类加 `override`

### 6. GRU 重置门 (reset gate) 未实现
- **文件**: `NPCore/Layers/Recurrent/GRU.cpp`
- **问题**: Forward 中 `n = tanh(n_in)` 完全没有使用 `r_t ⊙ (W_hn · h_{t-1})`，standard GRU 公式被简化
- **修复**:
  - Forward: 计算 `n_correction = Σ_j W_nh[i,j] * (r_j - 1) * h_prev[j]`，修正 n_t 的 net 输入
  - Backward: 正确的梯度链——reset gate 通过 n 间接影响 h_t；h_prev 梯度包含 z 门直接路径和 n 修正间接路径

### 7. SequenceLoader 构造函数参数被忽略
- **文件**: `NPCore/DataLoader.h`, `NPCore/DataLoader.cpp`
- **问题**: `SequenceLoader(int input_dim, int output_dim, int bs)` 中前两个参数完全忽略
- **修复**: 存储 `input_dim_`/`output_dim_`，`add_sequence()` 时校验维度（参数为 0 时跳过硬检查）

---

## 新增功能

### 8. Loss 函数补全
- **`cross_entropy_loss()`** (值): 之前只有 `cross_entropy_loss_grad()`（梯度），现在完整的 log-softmax + NLL loss
- **`bce_loss_grad()`** (梯度): 之前只有 `bce_loss()`（值）
- **`smooth_l1_loss_grad()`** (梯度): Huber loss 梯度

### 9. `loss_val()` 修复
- **文件**: `NPCore/Model.cpp`
- **问题**: 之前永远返回 `mse_loss`，无视 `LossType` 参数
- **修复**: → 当 `LossType == CrossEntropy` 时调用 `cross_entropy_loss()`

### 10. PositionalEncoding 层
- **文件**: `NPCore/Layers/Attention/PositionalEncoding.h/.cpp`（新增）
- **功能**: 正弦/余弦位置编码，Transformer 必需组件
- **用法**: `new PositionalEncoding(d_model, max_len)` → forward 输出 `x + PE`
- **无学习参数**: gradient 直接透传

### 11. Optimizer factory 函数扩展到 8 个
新增 `Momentum(lr)`, `Adagrad(lr)`, `Adadelta(lr)`, `NAdam(lr)`, `RAdam(lr)`
