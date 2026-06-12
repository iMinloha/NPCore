#ifndef COREPP_DATALOADER_H
#define COREPP_DATALOADER_H
#include <vector>
#include <functional>
#include "Core/Matrix.hpp"

namespace CoreNNSpace {

// =================================[DataLoader 抽象基类]================================
// 用户继承此类实现自己的数据加载逻辑。
//
// 数据格式约定:
//   每个样本 = (输入矩阵, 目标矩阵)
//   输入矩阵: FNN 用 (features, 1) 列向量
//            CNN 用 (H, W, C) 3D tensor
//            RNN 用 (seq_len, features) 每行一个时间步
//   目标矩阵: shape 与网络输出一致
//
//   批次: next_batch() 将多个样本在 row 维度拼接
//         FNN: (batch_size * features, 1) 或 (batch_size, features)
//         RNN: 仍需保持 (seq_len, features) — 不支持不等长序列批处理

enum class DataSplit { Train, Test, Val };

class DataLoader {
public:
    virtual ~DataLoader() = default;

    // --- 必须实现 ---
    virtual int  num_samples() const = 0;       // 当前 split 的样本总数
    virtual bool next_batch(Matrix<float>& x,   // [out] 输入批次, 由 DataLoader 填充
                            Matrix<float>& y) = 0; // [out] 目标批次
    // 返回 true 表示还有数据, false 表示 epoch 结束 (自动 reset)

    // --- 可选重写 ---
    virtual void reset() {}                     // 重置迭代器 (每个 epoch 开始时调用)
    virtual void set_split(DataSplit s) {}      // 切换 Train/Test/Val

    // --- 便捷方法 ---
    void train() { set_split(DataSplit::Train); }
    void test()  { set_split(DataSplit::Test);  }
    void val()   { set_split(DataSplit::Val);   }
};

// =================================[内置: 单样本 DataLoader]================================
// 用于只有一个 (x, y) 对的场景 — 每个 epoch 返回同一个样本 batch_size 次
class SingleSampleLoader : public DataLoader {
    Matrix<float> x_, y_;
    int batch_size_, count_ = 0;
public:
    SingleSampleLoader(const Matrix<float>& x, const Matrix<float>& y, int bs = 1)
        : x_(x), y_(y), batch_size_(bs) {}

    int  num_samples() const override { return 1; }
    void reset() override { count_ = 0; }

    bool next_batch(Matrix<float>& x, Matrix<float>& y) override {
        if (count_ >= batch_size_) { count_ = 0; return false; }
        x = x_; y = y_; count_++; return true;
    }
};

// =================================[内置: 内存 DataLoader]================================
// 从 std::vector<Matrix<float>> 加载, 自动划分 train/test/val
class InMemoryLoader : public DataLoader {
    std::vector<Matrix<float>> inputs_, targets_;
    std::vector<int> train_idx_, test_idx_, val_idx_;
    DataSplit split_ = DataSplit::Train;
    int batch_size_, cursor_ = 0;
    bool shuffle_ = true;

public:
    InMemoryLoader(int batch_size = 32, bool shuffle = true)
        : batch_size_(batch_size), shuffle_(shuffle) {}

    // 添加一个样本
    void add_sample(const Matrix<float>& x, const Matrix<float>& y) {
        inputs_.push_back(x); targets_.push_back(y);
    }

    // 按比例划分: train_ratio + test_ratio + val_ratio = 1.0
    void split(float train_r, float test_r, float val_r = 0.0f) {
        int n = (int)inputs_.size();
        train_idx_.clear(); test_idx_.clear(); val_idx_.clear();
        for (int i = 0; i < n; ++i) {
            if (i < n * train_r) train_idx_.push_back(i);
            else if (i < n * (train_r + test_r)) test_idx_.push_back(i);
            else val_idx_.push_back(i);
        }
    }

    int  num_samples() const override;
    void set_split(DataSplit s) override { split_ = s; cursor_ = 0; }
    void reset() override { cursor_ = 0; }
    bool next_batch(Matrix<float>& x, Matrix<float>& y) override;
};

} // namespace
#endif
