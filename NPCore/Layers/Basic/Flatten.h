#ifndef NPCORE_LAYERS_FLATTEN_H
#define NPCORE_LAYERS_FLATTEN_H

#include <vector>
#include "Layers/Module.h"

namespace NPCore {

// =================================[Flatten Layer]================================
// Flattens multi-dimensional input to a 2D column vector (col=1), for Conv2d -> Linear

class NPCORE_API Flatten : public Module<float> {
private:
    int original_row, original_col, original_channel;

public:
    Flatten() = default;
    ~Flatten() override = default;

    Matrix<float> forward(Matrix<float> &input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;

    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;
    void CleanGard() override;
};

} // namespace NPCore

#endif // NPCORE_LAYERS_FLATTEN_H
