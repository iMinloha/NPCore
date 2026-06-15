#include "Layers/Attention/PositionalEncoding.h"
#include <cmath>

namespace NPCore {

PositionalEncoding::PositionalEncoding(int d_model, float scale)
    : d_model(d_model), scale_(scale) {}

Matrix<float> PositionalEncoding::forward(Matrix<float>& input) {
    int seq_len = input.row;
    NPCORE_ASSERT(input.col == d_model, "PositionalEncoding: input dim mismatch");

    auto* out = new Matrix<float>(seq_len, d_model);

    // Compute positional encodings on the fly and add to input
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < d_model; ++i) {
            float angle = (float)pos / std::pow(10000.0f, (2.0f * (i / 2)) / d_model);
            float pe = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
            out->at(pos, i) = (input.at(pos, i) + pe) * scale_;
        }
    }

    output.push_back(out);
    return *out;
}

Matrix<float> PositionalEncoding::backward(Matrix<float>& grad_output) {
    // Gradients pass through unchanged (additive layer = identity gradient)
    auto* gi = new Matrix<float>(grad_output);
    if (scale_ != 1.0f) {
        int n = grad_output.row * grad_output.col;
        for (int i = 0; i < n; ++i)
            gi->data_ptr()[i] = grad_output.data_ptr()[i] * scale_;
    }
    return *gi;
}

void PositionalEncoding::CleanGard() {
    for (auto p : gard) { delete p; }
    gard.clear();
    for (auto p : output) { delete p; }
    output.clear();
}

std::vector<Matrix<float>*> PositionalEncoding::getParams() { return {}; }
Matrix<float>* PositionalEncoding::getGard() { return nullptr; }
Matrix<float>* PositionalEncoding::getOutput() { return output.empty() ? nullptr : output.back(); }

} // namespace NPCore
