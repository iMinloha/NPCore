#ifndef NPCORE_LAYERS_FLATTEN_H
#define NPCORE_LAYERS_FLATTEN_H

#include <vector>
#include "Layers/Module.h"

namespace NPCore {

// =================================[Flatten Layer]================================
// Flattens multi-dimensional input to a 2D column vector (col=1), for Conv2d -> Linear

class Flatten : public Module<float> {
private:
    int original_row, original_col, original_channel;

public:
    Flatten() = default;
    ~Flatten() override = default;  // gard, output cleaned by ~Module()

    Matrix<float> forward(Matrix<float> &input) override;

    // Backward: reshape flattened gradient back to original shape
    Matrix<float> backward(Matrix<float>& grad_output) override;

    std::vector<Matrix<float>*> getParams() override { return {}; }

    Matrix<float>* getGard() override { return gard.empty() ? nullptr : gard.back(); }

    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }

    void CleanGard() override {
        for (auto ptr : gard) delete ptr;
        gard.clear();
        for (auto ptr : output) delete ptr;
        output.clear();
    }
};

} // namespace NPCore

#endif // NPCORE_LAYERS_FLATTEN_H
