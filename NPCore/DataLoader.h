#ifndef NPCORE_DATALOADER_H
#define NPCORE_DATALOADER_H
#include <vector>
#include <string>
#include <functional>
#include "Core/Matrix.hpp"

namespace NPCore {

// =================================[DataLoader Abstract Base]================================
enum class DataSplit { Train, Test, Val };

class DataLoader {
public:
    virtual ~DataLoader() = default;

    virtual int  num_samples() const = 0;
    virtual bool next_batch(Matrix<float>& x, Matrix<float>& y) = 0;

    virtual void reset();
    virtual void set_split(DataSplit split);

    void train();
    void test();
    void val();
};

// =================================[SingleSampleLoader]================================
class SingleSampleLoader : public DataLoader {
    Matrix<float> x_, y_;
    int batch_size_, count_ = 0;
public:
    SingleSampleLoader(const Matrix<float>& x, const Matrix<float>& y, int bs = 1);
    int  num_samples() const override;
    void reset() override;
    bool next_batch(Matrix<float>& x, Matrix<float>& y) override;
};

// =================================[InMemoryLoader]================================
class InMemoryLoader : public DataLoader {
    std::vector<Matrix<float>> inputs_, targets_;
    std::vector<int> train_idx_, test_idx_, val_idx_;
    DataSplit split_ = DataSplit::Train;
    int batch_size_, cursor_ = 0;
    bool shuffle_ = true;

public:
    InMemoryLoader(int batch_size = 1, bool shuffle = true);
    void add_sample(const Matrix<float>& x, const Matrix<float>& y);
    void split(float train_r, float test_r, float val_r = 0.0f);
    int  num_samples() const override;
    void set_split(DataSplit s) override;
    void reset() override;
    bool next_batch(Matrix<float>& x, Matrix<float>& y) override;
};

// =================================[BatchStackLoader]================================
class BatchStackLoader : public DataLoader {
    DataLoader* source_;
    int batch_size_;
    bool pad_sequences_;
    std::vector<int> lengths_;

public:
    BatchStackLoader(DataLoader* src, int bs = 32, bool pad_seq = false);
    int  num_samples() const override;
    void set_split(DataSplit s) override;
    void reset() override;
    bool next_batch(Matrix<float>& x, Matrix<float>& y) override;
    const std::vector<int>& last_lengths() const;
};

// =================================[CSVLoader]================================
class CSVLoader : public DataLoader {
    std::string filepath_;
    int n_target_cols_;
    bool has_header_;
    std::vector<Matrix<float>> inputs_, targets_;
    std::vector<int> train_idx_, test_idx_, val_idx_;
    DataSplit split_ = DataSplit::Train;
    int cursor_ = 0;

public:
    CSVLoader(const std::string& path, int target_cols = 1, bool header = false);
    bool load();
    void parse(const std::string& csv_text);
    void add_row(const std::vector<float>& features, const std::vector<float>& targets);
    void split(float train_r, float test_r, float val_r = 0.0f);
    int  num_samples() const override;
    void set_split(DataSplit s) override;
    void reset() override;
    bool next_batch(Matrix<float>& x, Matrix<float>& y) override;
};

// =================================[ImageFolderLoader]================================
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
    int scan();
    void split(float train_r, float test_r, float val_r = 0.0f);
    int  num_samples() const override;
    int  num_classes() const;
    const std::vector<std::string>& classes() const;
    void set_split(DataSplit s) override;
    void reset() override;
    bool next_batch(Matrix<float>& x, Matrix<float>& y) override;
};

// =================================[SequenceLoader]================================
class SequenceLoader : public DataLoader {
    std::vector<Matrix<float>> inputs_, targets_;
    std::vector<int> train_idx_, test_idx_, val_idx_;
    DataSplit split_ = DataSplit::Train;
    int batch_size_, cursor_ = 0;

public:
    SequenceLoader(int input_dim = 0, int output_dim = 0, int bs = 16);
    void add_sequence(const Matrix<float>& x, const Matrix<float>& y);
    void split(float train_r, float test_r, float val_r = 0.0f);
    int  num_samples() const override;
    void set_split(DataSplit s) override;
    void reset() override;
    bool next_batch(Matrix<float>& x, Matrix<float>& y) override;
    bool next_batch(Matrix<float>& x, Matrix<float>& y, Matrix<float>& mask);
};

// =================================[Utility: Split rows of CSV text]================================
namespace detail {
    std::vector<std::string> split_csv_line(const std::string& line);
}

} // namespace NPCore
#endif
