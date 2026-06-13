#include "Layers/Basic/Flatten.h"

namespace NPCore {

Matrix<float> Flatten::forward(Matrix<float> &input) {
    // Save original shape for backward
    original_row = input.row;
    original_col = input.col;
    original_channel = input.channel;

    // Record input (stores original shape info)
    gard.push_back(new Matrix<float>(input));

    // Flatten: (row, col, channel) -> (row * col * channel, 1)
    int total = input.row * input.col * input.channel;
    auto* result = new Matrix<float>(total, 1);

    int idx = 0;
    for (int ch = 0; ch < input.channel; ++ch)
        for (int i = 0; i < input.row; ++i)
            for (int j = 0; j < input.col; ++j)
                result->at(idx++, 0) = input.at(i, j, ch);

    output.push_back(result);
    return *result;
}

Matrix<float> Flatten::backward(Matrix<float>& grad_output) {
    // Restore flattened gradient to original multi-dimensional shape
    auto* reshaped = new Matrix<float>(original_row, original_col, original_channel);

    int idx = 0;
    for (int ch = 0; ch < original_channel; ++ch)
        for (int i = 0; i < original_row; ++i)
            for (int j = 0; j < original_col; ++j)
                reshaped->at(i, j, ch) = grad_output.at(idx++, 0);

    return *reshaped;
}

} // namespace NPCore
