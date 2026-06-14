#include "Optimizers/Optimizer.h"
#include "Layers/Module.h"
#include <cmath>

namespace NPCore {

void SGD_method(std::vector<Module<float>*> params, Matrix<float>& grad, float lr) {
    Matrix<float> grad_ = grad;
    for (int i = params.size()-1; i >= 0; i--) {
        auto* l = params[i]; grad_ = l->backward(grad_);
        if (l->getParams().empty()) { l->CleanGard(); continue; }
        auto pl = l->getParams(), gl = l->getAllGrads();
        for (size_t p = 0; p < pl.size() && p < gl.size(); ++p)
            if (gl[p]) *pl[p] = *pl[p] - (*gl[p]) * lr;
        l->CleanGard();
    }
}

void Momentum_method(std::vector<Module<float>*> params, Matrix<float>& grad,
                     float lr, OptimState& st, float m) {
    Matrix<float> grad_ = grad;
    auto& mv = st.momentum_velocity;
    for (int i = params.size()-1; i >= 0; --i) {
        auto* l = params[i]; grad_ = l->backward(grad_);
        if (l->getParams().empty()) { l->CleanGard(); continue; }
        auto pl = l->getParams(), gl = l->getAllGrads();
        for (size_t p = 0; p < pl.size() && p < gl.size(); ++p) {
            auto* P = pl[p], *G = gl[p]; if (!G) continue;
            if (mv.find(P) == mv.end()) mv[P] = Matrix<float>(P->row, P->col);
            mv[P] = mv[P] * m + (*G) * lr; *P = *P - mv[P];
        }
        l->CleanGard();
    }
}

void Adam_method(std::vector<Module<float>*> params, Matrix<float>& grad, float lr,
                 OptimState& st, float b1, float b2, float eps) {
    st.adam_t++; int t = st.adam_t;
    float mc = 1/(1-std::pow(b1,t)), vc = 1/(1-std::pow(b2,t));
    Matrix<float> grad_ = grad;
    for (int i = params.size()-1; i >= 0; i--) {
        auto* l = params[i]; grad_ = l->backward(grad_);
        if (l->getParams().empty()) { l->CleanGard(); continue; }
        auto pl = l->getParams(), gl = l->getAllGrads();
        if (l->m.row == 0 && !pl.empty()) {
            l->m = Matrix<float>(pl[0]->row, pl[0]->col);
            l->v = Matrix<float>(pl[0]->row, pl[0]->col);
        }
        for (size_t p = 0; p < pl.size() && p < gl.size(); ++p) {
            auto* P = pl[p], *G = gl[p]; if (!G) continue;
            if (p == 0 && P->row == l->m.row && P->col == l->m.col)
                P->fused_adam_update(*G, l->m, l->v, b1, b2, mc, vc, lr, eps);
            else *P = *P - (*G) * lr;
        }
        l->CleanGard();
    }
}

void RMSProp_method(std::vector<Module<float>*> params, Matrix<float>& grad, float lr,
                    float beta, float eps) {
    Matrix<float> grad_ = grad;
    for (int i = params.size()-1; i >= 0; i--) {
        auto* l = params[i]; grad_ = l->backward(grad_);
        if (l->getParams().empty()) { l->CleanGard(); continue; }
        auto pl = l->getParams(), gl = l->getAllGrads();
        if (l->v.row == 0 && !pl.empty()) l->v = Matrix<float>(pl[0]->row, pl[0]->col);
        for (size_t p = 0; p < pl.size() && p < gl.size(); ++p) {
            auto* P = pl[p], *G = gl[p]; if (!G) continue;
            if (p == 0 && P->row == l->v.row && P->col == l->v.col)
                P->fused_rmsprop_update(*G, l->v, beta, lr, eps);
            else *P = *P - (*G) * lr;
        }
        l->CleanGard();
    }
}

void Adagrad_method(std::vector<Module<float>*> params, Matrix<float>& grad, float lr,
                    float eps) {
    Matrix<float> grad_ = grad;
    for (int i = params.size()-1; i >= 0; i--) {
        auto* l = params[i]; grad_ = l->backward(grad_);
        if (l->getParams().empty()) { l->CleanGard(); continue; }
        auto pl = l->getParams(), gl = l->getAllGrads();
        if (l->v.row == 0 && !pl.empty()) l->v = Matrix<float>(pl[0]->row, pl[0]->col);
        for (size_t p = 0; p < pl.size() && p < gl.size(); ++p) {
            auto* P = pl[p], *G = gl[p]; if (!G) continue;
            if (p == 0 && P->row == l->v.row && P->col == l->v.col)
                P->fused_adagrad_update(*G, l->v, lr, eps);
            else *P = *P - (*G) * lr;
        }
        l->CleanGard();
    }
}

void Adadelta_method(std::vector<Module<float>*> params, Matrix<float>& grad,
                     float rho, float eps) {
    Matrix<float> grad_ = grad;
    for (int i = params.size()-1; i >= 0; i--) {
        auto* l = params[i]; grad_ = l->backward(grad_);
        if (l->getParams().empty()) { l->CleanGard(); continue; }
        auto pl = l->getParams(), gl = l->getAllGrads();
        if (l->v.row == 0 && !pl.empty()) {
            l->v = Matrix<float>(pl[0]->row, pl[0]->col);
            l->m = Matrix<float>(pl[0]->row, pl[0]->col);
        }
        for (size_t p = 0; p < pl.size() && p < gl.size(); ++p) {
            auto* P = pl[p], *G = gl[p]; if (!G) continue;
            int n = P->row * P->col;
            for (int j = 0; j < n; ++j) {
                float g = G->data_ptr()[j];
                l->v.data_ptr()[j] = rho * l->v.data_ptr()[j] + (1-rho) * g * g;
                float dx = -std::sqrt(l->m.data_ptr()[j] + eps) / std::sqrt(l->v.data_ptr()[j] + eps) * g;
                l->m.data_ptr()[j] = rho * l->m.data_ptr()[j] + (1-rho) * dx * dx;
                P->data_ptr()[j] += dx;
            }
        }
        l->CleanGard();
    }
}

void NAdam_method(std::vector<Module<float>*> params, Matrix<float>& grad, float lr,
                  OptimState& st, float b1, float b2, float eps) {
    st.adam_t++; int t = st.adam_t;
    float mc = 1/(1-std::pow(b1,t)), vc = 1/(1-std::pow(b2,t));
    Matrix<float> grad_ = grad;
    for (int i = params.size()-1; i >= 0; i--) {
        auto* l = params[i]; grad_ = l->backward(grad_);
        if (l->getParams().empty()) { l->CleanGard(); continue; }
        auto pl = l->getParams(), gl = l->getAllGrads();
        if (l->m.row == 0 && !pl.empty()) {
            l->m = Matrix<float>(pl[0]->row, pl[0]->col);
            l->v = Matrix<float>(pl[0]->row, pl[0]->col);
        }
        for (size_t p = 0; p < pl.size() && p < gl.size(); ++p) {
            auto* P = pl[p], *G = gl[p]; if (!G) continue;
            int n = P->row * P->col;
            for (int j = 0; j < n; ++j) {
                float g = G->data_ptr()[j];
                l->m.data_ptr()[j] = b1 * l->m.data_ptr()[j] + (1-b1) * g;
                l->v.data_ptr()[j] = b2 * l->v.data_ptr()[j] + (1-b2) * g * g;
                float mh = l->m.data_ptr()[j] * mc;
                float vh = std::sqrt(l->v.data_ptr()[j] * vc) + eps;
                float mn = b1 * mh + (1-b1) * g / (1 - std::pow(b1, t));
                P->data_ptr()[j] -= lr * mn / vh;
            }
        }
        l->CleanGard();
    }
}

void RAdam_method(std::vector<Module<float>*> params, Matrix<float>& grad, float lr,
                  OptimState& st, float b1, float b2, float eps) {
    st.adam_t++; int t = st.adam_t;
    float ri = 2/(1-b2)-1;
    float rt = ri - 2*t*std::pow(b2,t)/(1-std::pow(b2,t));
    Matrix<float> grad_ = grad;
    for (int i = params.size()-1; i >= 0; i--) {
        auto* l = params[i]; grad_ = l->backward(grad_);
        if (l->getParams().empty()) { l->CleanGard(); continue; }
        auto pl = l->getParams(), gl = l->getAllGrads();
        if (l->m.row == 0 && !pl.empty()) {
            l->m = Matrix<float>(pl[0]->row, pl[0]->col);
            l->v = Matrix<float>(pl[0]->row, pl[0]->col);
        }
        for (size_t p = 0; p < pl.size() && p < gl.size(); ++p) {
            auto* P = pl[p], *G = gl[p]; if (!G) continue;
            int n = P->row * P->col;
            for (int j = 0; j < n; ++j) {
                float g = G->data_ptr()[j];
                l->m.data_ptr()[j] = b1 * l->m.data_ptr()[j] + (1-b1) * g;
                l->v.data_ptr()[j] = b2 * l->v.data_ptr()[j] + (1-b2) * g * g;
                float mh = l->m.data_ptr()[j] / (1 - std::pow(b1, t));
                if (rt > 4) {
                    float vh = std::sqrt(l->v.data_ptr()[j] / (1 - std::pow(b2, t))) + eps;
                    float r = std::sqrt((rt-4)*(rt-2)*ri / ((ri-4)*(ri-2)*rt));
                    P->data_ptr()[j] -= lr * r * mh / vh;
                } else {
                    P->data_ptr()[j] -= lr * mh;
                }
            }
        }
        l->CleanGard();
    }
}

// =================================[Wrapper functions — bind default hyperparameters]================================
void SGD_step(std::vector<Module<float>*> p, Matrix<float>& g, float lr, OptimState&) {
    SGD_method(p, g, lr);
}
void Momentum_step(std::vector<Module<float>*> p, Matrix<float>& g, float lr, OptimState& st) {
    Momentum_method(p, g, lr, st, 0.9f);
}
void Adam_step(std::vector<Module<float>*> p, Matrix<float>& g, float lr, OptimState& st) {
    Adam_method(p, g, lr, st, 0.9f, 0.999f, 1e-8f);
}
void RMSProp_step(std::vector<Module<float>*> p, Matrix<float>& g, float lr, OptimState&) {
    RMSProp_method(p, g, lr, 0.99f, 1e-8f);
}
void Adagrad_step(std::vector<Module<float>*> p, Matrix<float>& g, float lr, OptimState&) {
    Adagrad_method(p, g, lr, 1e-8f);
}
void Adadelta_step(std::vector<Module<float>*> p, Matrix<float>& g, float, OptimState&) {
    Adadelta_method(p, g, 0.9f, 1e-6f);
}
void NAdam_step(std::vector<Module<float>*> p, Matrix<float>& g, float lr, OptimState& st) {
    NAdam_method(p, g, lr, st, 0.9f, 0.999f, 1e-8f);
}
void RAdam_step(std::vector<Module<float>*> p, Matrix<float>& g, float lr, OptimState& st) {
    RAdam_method(p, g, lr, st, 0.9f, 0.999f, 1e-8f);
}

} // namespace NPCore
