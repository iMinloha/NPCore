#include "Layers/Architecture/Residual.h"

namespace NPCore {

Residual::Residual(Module<float>* layer, bool own) : sublayer(layer), owns_sublayer(own) {}
Residual::~Residual() { if (owns_sublayer) delete sublayer; }

Matrix<float> Residual::forward(Matrix<float>& input) {
    auto out = sublayer->forward(input);
    out = out + input;
    auto* saved = new Matrix<float>(out);
    output.push_back(saved);
    return *saved;
}

Matrix<float> Residual::backward(Matrix<float>& grad_output) {
    auto sub_grad = sublayer->backward(grad_output);
    return sub_grad + grad_output;
}

std::vector<Matrix<float>*> Residual::getParams() { return sublayer->getParams(); }
std::vector<Matrix<float>*> Residual::getAllGrads() { return sublayer->getAllGrads(); }

Matrix<float>* Residual::getGard() { return nullptr; }
Matrix<float>* Residual::getOutput() { return output.empty() ? nullptr : output.back(); }

void Residual::cuda() { sublayer->cuda(); }
void Residual::cpu()  { sublayer->cpu();  }
void Residual::eval()  { train_mode = false; sublayer->eval();  }
void Residual::train() { train_mode = true;  sublayer->train(); }

void Residual::CleanGard() {
    for (auto p : gard) { delete p; }
    gard.clear();
    for (auto p : output) { delete p; }
    output.clear();
    sublayer->CleanGard();
}

} // namespace NPCore
