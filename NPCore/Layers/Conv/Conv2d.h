#ifndef NPCORE_LAYERS_CONV2D_H
#define NPCORE_LAYERS_CONV2D_H

#include <vector>
#include <cmath>
#include "Layers/Module.h"
#include "Layers/ParamInit.h"
#include "Core/ConvUtils.h"

namespace NPCore {

class Conv2d : public Module<float> {
private:
    Matrix<float>* weight;
    Matrix<float>* bias;
    Matrix<float>* col_cache = nullptr;

    int in_channels, out_channels, kernel_size, stride, padding;

    // Cached weight_2d (lazily recomputed when dirty)
    mutable Matrix<float>* weight_2d_cache_ = nullptr;
    mutable bool weight_2d_dirty_ = true;

    Matrix<float> weight_2d() const;
    Matrix<float> grad_output_2d(const Matrix<float>& grad_output) const;

public:
    Conv2d(int in_channels, int out_channels, int kernel_size = 3,
           int stride = 1, int padding = 0,
           InitMode mode = InitMode::XavierUniform,
           double mu = 0.0, double sigma = 1.0);

    ~Conv2d() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;

    std::vector<Matrix<float>*> getParams() override;
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;

    void CleanGard() override;
};

} // namespace NPCore

#endif
