#ifndef NPCORE_LAYERS_RESIDUAL_H
#define NPCORE_LAYERS_RESIDUAL_H

#include "Layers/Module.h"

namespace NPCore {

// =================================[Residual Connection]================================
// y = F(x) + x — skip connection wrapper

class Residual : public Module<float> {
    Module<float>* sublayer;
    bool owns_sublayer;

public:
    Residual(Module<float>* layer, bool own = true);
    ~Residual() override;

    Matrix<float> forward(Matrix<float>& input) override;
    Matrix<float> backward(Matrix<float>& grad_output) override;
    void CleanGard() override;

    std::vector<Matrix<float>*> getParams() override;
    std::vector<Matrix<float>*> getAllGrads() override;
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty() ? nullptr : output.back(); }

    void cuda() override { sublayer->cuda(); }
    void cpu()  override { sublayer->cpu();  }
    void eval()  { train_mode = false; sublayer->eval();  }
    void train() { train_mode = true;  sublayer->train(); }
};

} // namespace NPCore
#endif
