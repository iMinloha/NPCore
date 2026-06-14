#include "Layers/Basic/Dropout.h"
#include "Core/RandomGenerator.h"

namespace NPCore {

Dropout::Dropout(float rate) : rate(rate) {}
Dropout::~Dropout() { delete mask_; }

Matrix<float> Dropout::forward(Matrix<float>& input) {
    auto* out = new Matrix<float>(input.row, input.col, input.channel);
    if (train_mode) {
        mask_ = new Matrix<float>(input.row, input.col, input.channel);
        float scale = 1.0f / (1.0f - rate);
        int n = input.row * input.col * input.channel;
        for (int i = 0; i < n; ++i) {
            float m = (RandomGenerator::uniform<float>(0,1) > rate) ? scale : 0.0f;
            mask_->data_ptr()[i] = m;
            out->data_ptr()[i] = input.data_ptr()[i] * m;
        }
    } else {
        int n = input.row * input.col * input.channel;
        for (int i = 0; i < n; ++i) out->data_ptr()[i] = input.data_ptr()[i];
    }
    output.push_back(out); return *out;
}

Matrix<float> Dropout::backward(Matrix<float>& grad_output) {
    auto* gi = new Matrix<float>(grad_output.row, grad_output.col, grad_output.channel);
    int n = grad_output.row * grad_output.col * grad_output.channel;
    for (int i = 0; i < n; ++i) gi->data_ptr()[i] = grad_output.data_ptr()[i] * mask_->data_ptr()[i];
    return *gi;
}

void Dropout::CleanGard() {
    for (auto p : gard) { delete p; }
    gard.clear();
    for (auto p : output) { delete p; }
    output.clear();
    delete mask_; mask_=nullptr;
}

std::vector<Matrix<float>*> Dropout::getParams() { return {}; }
Matrix<float>* Dropout::getGard() { return nullptr; }
Matrix<float>* Dropout::getOutput() { return output.empty() ? nullptr : output.back(); }

} // namespace NPCore
