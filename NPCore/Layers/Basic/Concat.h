#ifndef NPCORE_LAYERS_CONCAT_H
#define NPCORE_LAYERS_CONCAT_H

#include <vector>
#include "Layers/Module.h"

namespace NPCore {

// =================================[Concat — Branch Merging]================================
// Runs multiple sub-modules on the same input, concatenates outputs.
//
// For 2D input (features, 1): concat along rows → (Σfeatures, 1)
// For 3D input (H, W, C):     concat along channels → (H, W, ΣC)
//
// Backward: splits the gradient back to each branch.
//
// Usage:
//   Concat inception({new Conv2d(...), new Conv2d(...), new MaxPool2d(...)});
//   auto out = inception.forward(x);  // runs all branches, concats results

class NPCORE_API Concat : public Module<float> {
    std::vector<Module<float>*> branches;
    std::vector<int> branch_sizes;  // output size per branch (row count or channel count)

public:
    // Takes ownership of branches
    explicit Concat(std::vector<Module<float>*> branches);
    ~Concat() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override;
    std::vector<Matrix<float>*> getAllGrads() override;
    std::vector<Module<float>*> modules() override;
    const std::vector<Module<float>*>& getBranches() const { return branches; }
    Matrix<float>* getGard() override;
    Matrix<float>* getOutput() override;

    void cuda() override;
    void cpu() override;
    void eval() override;
    void train() override;
};

} // namespace NPCore

#endif
