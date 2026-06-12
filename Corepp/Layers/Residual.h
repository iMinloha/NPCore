#ifndef COREPP_LAYERS_RESIDUAL_H
#define COREPP_LAYERS_RESIDUAL_H
#include "Layers/Module.h"
namespace CoreNNSpace {

// =================================[Residual Connection]================================
// y = F(x) + x  (skip connection wrapper)
// Wraps any layer and adds its input to its output.
// Shape must be preserved through the wrapped layer.

class Residual : public Module<float> {
    Module<float>* sublayer;
    bool owns_sublayer;
public:
    Residual(Module<float>* layer, bool own = true) : sublayer(layer), owns_sublayer(own) {}
    ~Residual() override { if (owns_sublayer) delete sublayer; }

    Matrix<float> forward(Matrix<float>& input) override {
        auto out = sublayer->forward(input);
        out = out + input;  // skip connection
        auto* saved = new Matrix<float>(out);
        output.push_back(saved);
        return *saved;
    }

    Matrix<float> backward(Matrix<float>& grad_output) override {
        auto sub_grad = sublayer->backward(grad_output);
        return sub_grad + grad_output;  // gradient flows through both paths
    }

    std::vector<Matrix<float>*> getParams() override { return sublayer->getParams(); }
    std::vector<Matrix<float>*> getAllGrads() override { return sublayer->getAllGrads(); }
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }

    void cuda() override { sublayer->cuda(); }
    void cpu()  override { sublayer->cpu();  }
    void eval()  { train_mode=false; sublayer->eval();  }
    void train() { train_mode=true;  sublayer->train(); }

    void CleanGard() override {
        for (auto p : gard) delete p;
        gard.clear();
        for (auto p : output) delete p;
        output.clear();
        sublayer->CleanGard();
    }
};
} // namespace
#endif
