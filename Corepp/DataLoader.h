#ifndef COREPP_DATALOADER_H
#define COREPP_DATALOADER_H
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <random>
#include <sstream>
#include <fstream>
#include "Core/Matrix.hpp"

namespace CoreNNSpace {

// =================================[DataLoader Abstract Base]================================
// Users inherit this class to implement their own data loading logic.
//
// Data format conventions:
//   Each sample = (input matrix, target matrix)
//   Input:  FNN uses (features, 1) column vector
//           CNN uses (H, W, C) 3D tensor
//           RNN uses (seq_len, features) -- one row per time step
//   Target: shape matches network output
//
//   Batching: next_batch() stacks multiple samples along the row dimension.
//             FNN: (batch_size, features) row-stacked
//             CNN: (batch_size, H*W*C) or individual 3D tensors
//             RNN: padded to max seq_len, (max_len, features * batch_size)

enum class DataSplit { Train, Test, Val };

class DataLoader {
public:
    virtual ~DataLoader() = default;

    // --- Required ---
    virtual int  num_samples() const = 0;
    virtual bool next_batch(Matrix<float>& x, Matrix<float>& y) = 0;
    // Returns true if more data available, false when epoch ends (auto reset)

    // --- Optional overrides ---
    virtual void reset() {}
    virtual void set_split(DataSplit /*split*/) {}

    // --- Convenience ---
    void train() { set_split(DataSplit::Train); }
    void test()  { set_split(DataSplit::Test);  }
    void val()   { set_split(DataSplit::Val);   }
};

// =================================[SingleSampleLoader]================================
// For single (x, y) pair -- returns the same sample batch_size times per epoch.
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

// =================================[InMemoryLoader]================================
// Vector-based loader with train/test/val split and optional shuffling.
// Samples are returned one-at-a-time (batch_size=1) for fine-grained control.
class InMemoryLoader : public DataLoader {
    std::vector<Matrix<float>> inputs_, targets_;
    std::vector<int> train_idx_, test_idx_, val_idx_;
    DataSplit split_ = DataSplit::Train;
    int batch_size_, cursor_ = 0;
    bool shuffle_ = true;

public:
    InMemoryLoader(int batch_size = 1, bool shuffle = true)
        : batch_size_(batch_size), shuffle_(shuffle) {}

    // Add one sample
    void add_sample(const Matrix<float>& x, const Matrix<float>& y) {
        inputs_.push_back(x); targets_.push_back(y);
    }

    // Split ratios: train_r + test_r + val_r = 1.0
    void split(float train_r, float test_r, float /*val_r*/ = 0.0f) {
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

// =================================[BatchStackLoader -- Proper Batching]================================
// Wraps any DataLoader and stacks individual samples into batches.
// For FNN: (features,1) x batch_size -> (batch_size, features) row-stacked.
// For CNN:  each sample keeps its shape, batch stored as vector (trainer handles iteration).
// For RNN:  pads variable-length sequences to max length in the batch.
//
// Usage:
//   InMemoryLoader raw;
//   raw.add_sample(...); raw.split(0.8, 0.2);
//   BatchStackLoader batched(raw, 32);  // stacks 32 samples per batch
//   batched.next_batch(x, y);  // x is (batch_size, features) for FNN

class BatchStackLoader : public DataLoader {
    DataLoader* source_;
    int batch_size_;
    bool pad_sequences_;       // for RNN: pad to max length in batch
    std::vector<int> lengths_; // original lengths before padding (for RNN unpack)

public:
    BatchStackLoader(DataLoader* src, int bs = 32, bool pad_seq = false)
        : source_(src), batch_size_(bs), pad_sequences_(pad_seq) {}

    int  num_samples() const override { return source_->num_samples() / batch_size_; }
    void set_split(DataSplit s) override { source_->set_split(s); }
    void reset() override { source_->reset(); }

    bool next_batch(Matrix<float>& x, Matrix<float>& y) override;
    const std::vector<int>& last_lengths() const { return lengths_; }
};

// =================================[CSVLoader -- Tabular Data]================================
// Loads feature-target pairs from CSV-formatted strings or files.
// Format: each row = feature_1, feature_2, ..., target_1, target_2, ...
// First n_target columns are targets, remaining are features.
//
// Usage:
//   CSVLoader csv("data.csv", /*n_target_cols=*/1, /*has_header=*/true);
//   csv.load();  // reads and parses the file
//   csv.split(0.7, 0.3);
//   csv.train();

class CSVLoader : public DataLoader {
    std::string filepath_;
    int n_target_cols_;
    bool has_header_;
    std::vector<Matrix<float>> inputs_, targets_;
    std::vector<int> train_idx_, test_idx_, val_idx_;
    DataSplit split_ = DataSplit::Train;
    int cursor_ = 0;

public:
    CSVLoader(const std::string& path, int target_cols = 1, bool header = false)
        : filepath_(path), n_target_cols_(target_cols), has_header_(header) {}

    // Parse the file. Returns true on success.
    bool load();

    // Parse from a string (for embedded data or stringstream)
    void parse(const std::string& csv_text);

    // Manually add a parsed row
    void add_row(const std::vector<float>& features, const std::vector<float>& targets);

    void split(float train_r, float test_r, float val_r = 0.0f);
    int  num_samples() const override;
    void set_split(DataSplit s) override { split_ = s; cursor_ = 0; }
    void reset() override { cursor_ = 0; }
    bool next_batch(Matrix<float>& x, Matrix<float>& y) override;
};

// =================================[ImageFolderLoader -- Image Data]================================
// Loads images from a directory. Images are stored as 3D tensors (H, W, C).
// User must provide a decode function that reads an image file into a Matrix.
//
// Directory structure:
//   root/
//     class_0/  image_001.png  image_002.png ...
//     class_1/  image_001.png  image_002.png ...
//
// Usage:
//   auto decode = [](const std::string& path) -> Matrix<float> {
//       // Use your image library (stb_image, OpenCV, etc.)
//       int w, h, c;
//       unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 0);
//       Matrix<float> img(h, w, c);
//       for (...) img.at(i,j,ch) = data[...] / 255.0f;
//       stbi_image_free(data);
//       return img;
//   };
//   ImageFolderLoader dataset("data/images", decode, /*batch_size=*/32);
//   dataset.split(0.8, 0.2);
//   dataset.train();

using ImageDecoder = std::function<Matrix<float>(const std::string& filepath)>;

class ImageFolderLoader : public DataLoader {
    std::string root_dir_;
    ImageDecoder decoder_;
    int batch_size_;
    std::vector<std::string> paths_;
    std::vector<int> labels_;
    std::vector<std::string> class_names_;
    std::vector<int> train_idx_, test_idx_, val_idx_;
    DataSplit split_ = DataSplit::Train;
    int cursor_ = 0;

public:
    ImageFolderLoader(const std::string& root, ImageDecoder decoder, int bs = 32);

    // Scan directory, populate paths and labels. Returns number of images found.
    int scan();

    void split(float train_r, float test_r, float val_r = 0.0f);
    int  num_samples() const override;
    int  num_classes() const { return (int)class_names_.size(); }
    const std::vector<std::string>& classes() const { return class_names_; }

    void set_split(DataSplit s) override { split_ = s; cursor_ = 0; }
    void reset() override { cursor_ = 0; }
    bool next_batch(Matrix<float>& x, Matrix<float>& y) override;
};

// =================================[SequenceLoader -- Variable-Length Sequences]================================
// Loads variable-length sequences with automatic padding to batch max.
// Each sample is (seq_len_i, features). Batches are padded to (max_len, features).
// A mask matrix indicates valid positions (1.0 = valid, 0.0 = padding).
//
// Usage:
//   SequenceLoader seq_loader(/*input_dim=*/10, /*output_dim=*/1);
//   seq_loader.add_sequence(input_seq, target_val);  // add one sequence
//   seq_loader.split(0.8, 0.2);
//   seq_loader.train();
//   Matrix<float> x, y, mask;
//   while (seq_loader.next_batch(x, y, mask)) { ... }

class SequenceLoader : public DataLoader {
    std::vector<Matrix<float>> inputs_, targets_;
    std::vector<int> train_idx_, test_idx_, val_idx_;
    DataSplit split_ = DataSplit::Train;
    int batch_size_, cursor_ = 0;

public:
    SequenceLoader(int /*input_dim*/ = 0, int /*output_dim*/ = 0, int bs = 16)
        : batch_size_(bs) {}

    // Add one sequence: input is (seq_len, features), target is (seq_len, out_features)
    void add_sequence(const Matrix<float>& x, const Matrix<float>& y) {
        inputs_.push_back(x); targets_.push_back(y);
    }

    void split(float train_r, float test_r, float val_r = 0.0f);

    int  num_samples() const override { return (int)inputs_.size(); }
    void set_split(DataSplit s) override { split_ = s; cursor_ = 0; }
    void reset() override { cursor_ = 0; }

    // Standard next_batch without mask (backward compatible)
    bool next_batch(Matrix<float>& x, Matrix<float>& y) override;

    // Extended: also returns a mask matrix (1.0=valid, 0.0=padding)
    bool next_batch(Matrix<float>& x, Matrix<float>& y, Matrix<float>& mask);
};

// =================================[Utility: Split rows of CSV text]================================
namespace detail {
    inline std::vector<std::string> split_csv_line(const std::string& line) {
        std::vector<std::string> result;
        std::string cell;
        bool in_quotes = false;
        for (char c : line) {
            if (c == '"') { in_quotes = !in_quotes; continue; }
            if (c == ',' && !in_quotes) {
                // trim trailing \r
                while (!cell.empty() && cell.back() == '\r') cell.pop_back();
                result.push_back(cell);
                cell.clear();
            } else {
                cell += c;
            }
        }
        while (!cell.empty() && cell.back() == '\r') cell.pop_back();
        if (!cell.empty() || !result.empty()) result.push_back(cell);
        return result;
    }
}

} // namespace CoreNNSpace
#endif
