#include "DataLoader.h"
#include <algorithm>
#include <random>
#include <fstream>
#include <filesystem>
#include <sstream>

namespace CoreNNSpace {

// =================================[InMemoryLoader]================================

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
    if (cursor_ >= total) {
        cursor_ = 0;
        if (shuffle_) {
            static std::mt19937 rng(42);
            std::shuffle(idx.begin(), idx.end(), rng);
        }
        return false;
    }

    if (batch_size_ <= 1) {
        // Single sample per call
        int i = idx[cursor_++];
        x = inputs_[i];
        y = targets_[i];
    } else {
        // Stack batch_size samples into one matrix along rows
        int remaining = total - cursor_;
        int n = std::min(batch_size_, remaining);
        int first = idx[cursor_];
        int feat_cols = inputs_[first].col;
        int targ_cols = targets_[first].col;

        // Allocate row-stacked output
        x = Matrix<float>(n, feat_cols);
        y = Matrix<float>(n, targ_cols);

        for (int k = 0; k < n; ++k) {
            int i = idx[cursor_++];
            for (int c = 0; c < feat_cols; ++c)
                x.at(k, c) = inputs_[i].at(0, c) * inputs_[i].row; // handle both (1,F) and (F,1)
            // Simple row copy: copy row 0 of each sample
            if (inputs_[i].row == 1) {
                for (int c = 0; c < feat_cols; ++c) x.at(k, c) = inputs_[i].at(0, c);
            } else {
                for (int r = 0; r < inputs_[i].row; ++r)
                    for (int c = 0; c < feat_cols; ++c)
                        x.at(k * inputs_[i].row + r, c) = inputs_[i].at(r, c);
            }
            for (int c = 0; c < targ_cols; ++c) y.at(k, c) = targets_[i].at(0, c);
        }
    }
    return true;
}

// =================================[BatchStackLoader]================================

bool BatchStackLoader::next_batch(Matrix<float>& x, Matrix<float>& y) {
    lengths_.clear();
    Matrix<float> sx, sy;
    int count = 0;
    int max_len = 0;

    // Collect batch_size samples
    std::vector<Matrix<float>> xs, ys;
    for (int i = 0; i < batch_size_; ++i) {
        if (!source_->next_batch(sx, sy)) break;
        xs.push_back(sx);
        ys.push_back(sy);
        if (pad_sequences_ && sx.row > max_len) max_len = sx.row;
        ++count;
    }
    if (count == 0) return false;

    if (!pad_sequences_) {
        // Simple stacking: all samples must have same shape
        int cols = xs[0].col;
        x = Matrix<float>(count, cols);
        y = Matrix<float>(count, ys[0].col);
        for (int i = 0; i < count; ++i) {
            for (int c = 0; c < cols; ++c) x.at(i, c) = xs[i].at(0, c);
            for (int c = 0; c < ys[i].col; ++c) y.at(i, c) = ys[i].at(0, c);
        }
    } else {
        // Pad sequences to max_len
        int feat = xs[0].col;
        int tout = ys[0].col;
        x = Matrix<float>(max_len, feat * count);
        y = Matrix<float>(max_len, tout * count);
        for (int i = 0; i < count; ++i) {
            int len = xs[i].row;
            lengths_.push_back(len);
            for (int t = 0; t < len; ++t) {
                for (int f = 0; f < feat; ++f)
                    x.at(t, i * feat + f) = xs[i].at(t, f);
                for (int o = 0; o < tout; ++o)
                    y.at(t, i * tout + o) = ys[i].at(t, o);
            }
            // Remaining rows stay zero (padding)
        }
    }
    return true;
}

// =================================[CSVLoader]================================

bool CSVLoader::load() {
    std::ifstream file(filepath_);
    if (!file.is_open()) return false;

    std::string line;
    if (has_header_) std::getline(file, line); // skip header

    std::string all;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        all += line + "\n";
    }
    parse(all);
    return true;
}

void CSVLoader::parse(const std::string& csv_text) {
    std::istringstream ss(csv_text);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        auto cells = detail::split_csv_line(line);
        if (cells.empty()) continue;

        // First n_target_cols are targets, rest are features
        int total = (int)cells.size();
        int n_feat = total - n_target_cols_;
        if (n_feat <= 0) continue;

        std::vector<float> feats, targs;
        for (int i = 0; i < n_target_cols_; ++i)
            targs.push_back(std::stof(cells[i]));
        for (int i = n_target_cols_; i < total; ++i)
            feats.push_back(std::stof(cells[i]));

        add_row(feats, targs);
    }
}

void CSVLoader::add_row(const std::vector<float>& features,
                        const std::vector<float>& targets) {
    Matrix<float> x((int)features.size(), 1);
    Matrix<float> y((int)targets.size(), 1);
    for (size_t i = 0; i < features.size(); ++i) x.at((int)i, 0) = features[i];
    for (size_t i = 0; i < targets.size(); ++i)  y.at((int)i, 0) = targets[i];
    inputs_.push_back(x);
    targets_.push_back(y);
}

void CSVLoader::split(float train_r, float test_r, float /*val_r*/) {
    int n = (int)inputs_.size();
    train_idx_.clear(); test_idx_.clear(); val_idx_.clear();
    for (int i = 0; i < n; ++i) {
        if (i < n * train_r) train_idx_.push_back(i);
        else if (i < n * (train_r + test_r)) test_idx_.push_back(i);
        else val_idx_.push_back(i);
    }
}

int CSVLoader::num_samples() const {
    if (split_ == DataSplit::Train) return (int)train_idx_.size();
    if (split_ == DataSplit::Test)  return (int)test_idx_.size();
    return (int)val_idx_.size();
}

bool CSVLoader::next_batch(Matrix<float>& x, Matrix<float>& y) {
    auto& idx = (split_ == DataSplit::Train) ? train_idx_
              : (split_ == DataSplit::Test)  ? test_idx_ : val_idx_;
    if (cursor_ >= (int)idx.size()) { cursor_ = 0; return false; }
    int i = idx[cursor_++];
    x = inputs_[i];
    y = targets_[i];
    return true;
}

// =================================[ImageFolderLoader]================================

ImageFolderLoader::ImageFolderLoader(const std::string& root, ImageDecoder decoder, int bs)
    : root_dir_(root), decoder_(decoder), batch_size_(bs) {}

int ImageFolderLoader::scan() {
    namespace fs = std::filesystem;
    paths_.clear();
    labels_.clear();
    class_names_.clear();

    if (!fs::exists(root_dir_) || !fs::is_directory(root_dir_)) return 0;

    for (const auto& entry : fs::directory_iterator(root_dir_)) {
        if (!entry.is_directory()) continue;
        std::string class_name = entry.path().filename().string();
        int label = (int)class_names_.size();
        class_names_.push_back(class_name);

        for (const auto& img_entry : fs::directory_iterator(entry.path())) {
            if (!img_entry.is_regular_file()) continue;
            std::string ext = img_entry.path().extension().string();
            // Common image extensions
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
                ext == ".bmp" || ext == ".tga" || ext == ".ppm") {
                paths_.push_back(img_entry.path().string());
                labels_.push_back(label);
            }
        }
    }
    return (int)paths_.size();
}

void ImageFolderLoader::split(float train_r, float test_r, float /*val_r*/) {
    int n = (int)paths_.size();
    train_idx_.clear(); test_idx_.clear(); val_idx_.clear();
    for (int i = 0; i < n; ++i) {
        if (i < n * train_r) train_idx_.push_back(i);
        else if (i < n * (train_r + test_r)) test_idx_.push_back(i);
        else val_idx_.push_back(i);
    }
}

int ImageFolderLoader::num_samples() const {
    if (split_ == DataSplit::Train) return (int)train_idx_.size();
    if (split_ == DataSplit::Test)  return (int)test_idx_.size();
    return (int)val_idx_.size();
}

bool ImageFolderLoader::next_batch(Matrix<float>& x, Matrix<float>& y) {
    auto& idx = (split_ == DataSplit::Train) ? train_idx_
              : (split_ == DataSplit::Test)  ? test_idx_ : val_idx_;
    if (cursor_ >= (int)idx.size()) { cursor_ = 0; return false; }

    // Decode one image at a time for simplicity
    // For production, batch decoding can be added
    int i = idx[cursor_++];
    x = decoder_(paths_[i]);

    // Target: one-hot encoded label
    int n_classes = (int)class_names_.size();
    y = Matrix<float>(n_classes, 1);
    for (int c = 0; c < n_classes; ++c) y.at(c, 0) = 0.0f;
    y.at(labels_[i], 0) = 1.0f;

    return true;
}

// =================================[SequenceLoader]================================

void SequenceLoader::split(float train_r, float test_r, float /*val_r*/) {
    int n = (int)inputs_.size();
    train_idx_.clear(); test_idx_.clear(); val_idx_.clear();
    for (int i = 0; i < n; ++i) {
        if (i < n * train_r) train_idx_.push_back(i);
        else if (i < n * (train_r + test_r)) test_idx_.push_back(i);
        else val_idx_.push_back(i);
    }
}

bool SequenceLoader::next_batch(Matrix<float>& x, Matrix<float>& y) {
    Matrix<float> dummy_mask;
    return next_batch(x, y, dummy_mask);
}

bool SequenceLoader::next_batch(Matrix<float>& x, Matrix<float>& y, Matrix<float>& mask) {
    auto& idx = (split_ == DataSplit::Train) ? train_idx_
              : (split_ == DataSplit::Test)  ? test_idx_ : val_idx_;
    if (cursor_ >= (int)idx.size()) { cursor_ = 0; return false; }

    // Collect batch_size sequences, find max length
    int remaining = (int)idx.size() - cursor_;
    int n = std::min(batch_size_, remaining);
    int max_len = 0;
    for (int k = 0; k < n; ++k) {
        int i = idx[cursor_ + k];
        if (inputs_[i].row > max_len) max_len = inputs_[i].row;
    }

    int feat = inputs_[idx[cursor_]].col;
    int tout = targets_[idx[cursor_]].col;

    // Allocate padded batch: (max_len, features * n) and (max_len, targets * n)
    x = Matrix<float>(max_len, feat * n);
    y = Matrix<float>(max_len, tout * n);
    mask = Matrix<float>(max_len, n);  // binary mask: 1.0 = valid, 0.0 = pad

    for (int k = 0; k < n; ++k) {
        int i = idx[cursor_++];
        int len = inputs_[i].row;
        for (int t = 0; t < max_len; ++t) {
            mask.at(t, k) = (t < len) ? 1.0f : 0.0f;
            if (t < len) {
                for (int f = 0; f < feat; ++f)
                    x.at(t, k * feat + f) = inputs_[i].at(t, f);
                for (int o = 0; o < tout; ++o)
                    y.at(t, k * tout + o) = targets_[i].at(t, o);
            }
        }
    }
    return true;
}

} // namespace CoreNNSpace
