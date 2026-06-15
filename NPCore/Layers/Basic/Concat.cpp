// =================================[Concat — Branch Merging]================================
#include "Layers/Basic/Concat.h"
#include "Core/Assert.h"

namespace NPCore {

Concat::Concat(std::vector<Module<float>*> branches) : branches(std::move(branches)) {}

Concat::~Concat() { for (auto* b : branches) delete b; }

Matrix<float> Concat::forward(Matrix<float>& input) {
    branch_sizes.clear();
    NPCORE_ASSERT(!branches.empty(), "Concat: need at least 1 branch");

    std::vector<Matrix<float>> outputs;
    for (auto* b : branches) outputs.push_back(b->forward(input));

    bool is_3d = (outputs[0].channel > 1);
    int total_size = 0;

    for (size_t i = 0; i < outputs.size(); ++i) {
        int sz = is_3d ? outputs[i].channel : outputs[i].row;
        branch_sizes.push_back(sz);
        total_size += sz;
    }

    // Allocate concatenated output
    if (is_3d) {
        int H = outputs[0].row, W = outputs[0].col;
        auto* result = new Matrix<float>(H, W, total_size);
        int ch_offset = 0;
        for (size_t i = 0; i < outputs.size(); ++i) {
            for (int c = 0; c < outputs[i].channel; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        result->at(h, w, ch_offset + c) = outputs[i].at(h, w, c);
            ch_offset += outputs[i].channel;
        }
        output.push_back(result);
        return *result;
    } else {
        auto* result = new Matrix<float>(total_size, 1);
        int row = 0;
        for (size_t i = 0; i < outputs.size(); ++i) {
            for (int r = 0; r < outputs[i].row; ++r, ++row)
                result->at(row, 0) = outputs[i].at(r, 0);
        }
        output.push_back(result);
        return *result;
    }
}

Matrix<float> Concat::backward(Matrix<float>& grad_output) {
    bool is_3d = (grad_output.channel > 1);
    std::vector<Matrix<float>> branch_grads;
    int offset = 0;

    for (size_t i = 0; i < branches.size(); ++i) {
        int sz = branch_sizes[i];
        if (is_3d) {
            int H = grad_output.row, W = grad_output.col;
            auto* gi = new Matrix<float>(H, W, sz);
            for (int c = 0; c < sz; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        gi->at(h, w, c) = grad_output.at(h, w, offset + c);
            branch_grads.push_back(*gi);
            offset += sz;
        } else {
            auto* gi = new Matrix<float>(sz, 1);
            for (int r = 0; r < sz; ++r)
                gi->at(r, 0) = grad_output.at(offset + r, 0);
            branch_grads.push_back(*gi);
            offset += sz;
        }
    }

    // Backward through each branch, sum the input gradients
    Matrix<float> d_input = branches[0]->backward(branch_grads[0]);
    for (size_t i = 1; i < branches.size(); ++i) {
        Matrix<float> d_i = branches[i]->backward(branch_grads[i]);
        d_input = d_input + d_i;  // gradients from all branches sum at the input
    }

    return d_input;
}

std::vector<Matrix<float>*> Concat::getParams() {
    std::vector<Matrix<float>*> all;
    for (auto* b : branches) {
        auto p = b->getParams();
        all.insert(all.end(), p.begin(), p.end());
    }
    return all;
}

std::vector<Matrix<float>*> Concat::getAllGrads() {
    std::vector<Matrix<float>*> all;
    for (auto* b : branches) {
        auto g = b->getAllGrads();
        all.insert(all.end(), g.begin(), g.end());
    }
    return all;
}

std::vector<Module<float>*> Concat::modules() {
    std::vector<Module<float>*> all;
    for (auto* b : branches) {
        auto sub = b->modules();
        all.insert(all.end(), sub.begin(), sub.end());
    }
    return all;
}

Matrix<float>* Concat::getGard() { return nullptr; }
Matrix<float>* Concat::getOutput() { return output.empty() ? nullptr : output.back(); }

void Concat::CleanGard() {
    for (auto* b : branches) b->CleanGard();
    for (auto p : gard)   { delete p; }
    for (auto p : output) { delete p; }
    gard.clear();
    output.clear();
}

void Concat::cuda()   { for (auto* b : branches) b->cuda();  }
void Concat::cpu()    { for (auto* b : branches) b->cpu();   }
void Concat::eval()   { train_mode = false; for (auto* b : branches) b->eval();  }
void Concat::train()  { train_mode = true;  for (auto* b : branches) b->train(); }

} // namespace NPCore
