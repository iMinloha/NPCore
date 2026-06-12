#include "DataLoader.h"
#include <algorithm>
#include <random>

namespace CoreNNSpace {

int InMemoryLoader::num_samples() const {
    if (split_ == DataSplit::Train) return (int)train_idx_.size();
    if (split_ == DataSplit::Test)  return (int)test_idx_.size();
    return (int)val_idx_.size();
}

bool InMemoryLoader::next_batch(Matrix<float>& x, Matrix<float>& y) {
    auto& idx = (split_ == DataSplit::Train) ? train_idx_
              : (split_ == DataSplit::Test)  ? test_idx_ : val_idx_;
    int total = (int)idx.size();
    if (total == 0) return false;
    if (cursor_ >= total) { cursor_ = 0; return false; }

    // 简单实现: 每次返回一个样本 (可扩展为真实 batch 拼接)
    int i = idx[cursor_++];
    x = inputs_[i];
    y = targets_[i];
    return true;
}

} // namespace
