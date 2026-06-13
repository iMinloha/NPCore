#ifndef COREPP_AUTOGRAD_H
#define COREPP_AUTOGRAD_H

// =================================[CorePP/Autograd - Gradient Checking Utility]================================
//
// Lightweight numerical gradient verification - NOT a full autodiff framework.
// Use these to verify that hand-derived analytical gradients are correct.
//
// Decision: Do NOT introduce automatic symbolic differentiation.
// Reason: The current operation set (Linear, Conv2d, activations) has well-known
// closed-form gradients. A full autodiff engine would require a complete rewrite
// of the library's architecture (compute graph, tape recording, operator overloading
// for gradient tracking). The ROI is low compared to optimizing the existing
// hand-derived gradients and adding numerical verification.
//
// Instead: use numerical_gradient() and gradcheck() below to validate gradients
// whenever a new layer or loss function is added.

#include <cmath>
#include <iostream>
#include <functional>
#include "Core/Matrix.hpp"

namespace CoreNNSpace {

// =================================[Numerical Gradient (Finite Differences)]================================
// Computes dL/dx at point x using central finite differences.
// f: loss function that takes a Matrix and returns a scalar
// x: point at which to compute gradient
// epsilon: perturbation size (default 1e-5)
template<typename T>
Matrix<T> numerical_gradient(std::function<T(const Matrix<T>&)> f,
                              const Matrix<T>& x, T epsilon = T(1e-5)) {
    Matrix<T> grad(x.row, x.col, x.channel);

    for (int ch = 0; ch < x.channel; ++ch) {
        for (int i = 0; i < x.row; ++i) {
            for (int j = 0; j < x.col; ++j) {
                T original = x.at(i, j, ch);

                // Central difference: (f(x+h) - f(x-h)) / (2h)
                Matrix<T> x_plus = x;
                x_plus.at(i, j, ch) = original + epsilon;
                T f_plus = f(x_plus);

                Matrix<T> x_minus = x;
                x_minus.at(i, j, ch) = original - epsilon;
                T f_minus = f(x_minus);

                grad.at(i, j, ch) = (f_plus - f_minus) / (T(2) * epsilon);
            }
        }
    }

    return grad;
}

// =================================[Gradient Check]================================
// Compares analytical gradient against numerical gradient.
// Returns true if the relative error is below threshold for all elements.
// Prints detailed comparison to stderr.
template<typename T>
bool gradcheck(const Matrix<T>& analytical_grad,
               const Matrix<T>& numerical_grad,
               T threshold = T(1e-3),
               const char* label = "") {
    if (analytical_grad.row != numerical_grad.row ||
        analytical_grad.col != numerical_grad.col ||
        analytical_grad.channel != numerical_grad.channel) {
        std::cerr << "[gradcheck] ERROR: shape mismatch - analytical ("
                  << analytical_grad.row << "," << analytical_grad.col << ","
                  << analytical_grad.channel << ") vs numerical ("
                  << numerical_grad.row << "," << numerical_grad.col << ","
                  << numerical_grad.channel << ")" << std::endl;
        return false;
    }

    T max_abs_error = 0;
    T max_rel_error = 0;
    int max_i = 0, max_j = 0, max_ch = 0;
    bool all_ok = true;

    for (int ch = 0; ch < analytical_grad.channel; ++ch) {
        for (int i = 0; i < analytical_grad.row; ++i) {
            for (int j = 0; j < analytical_grad.col; ++j) {
                T a = analytical_grad.at(i, j, ch);
                T n = numerical_grad.at(i, j, ch);

                T abs_err = std::abs(a - n);
                T denom = std::max(std::abs(a), std::abs(n));
                T rel_err = (denom > T(1e-9)) ? abs_err / denom : abs_err;

                if (abs_err > max_abs_error) {
                    max_abs_error = abs_err;
                    max_i = i; max_j = j; max_ch = ch;
                }
                if (rel_err > max_rel_error) {
                    max_rel_error = rel_err;
                }

                if (rel_err > threshold && abs_err > T(1e-7)) {
                    all_ok = false;
                }
            }
        }
    }

    if (all_ok) {
        std::cout << "[gradcheck] PASS" << (label[0] ? "-" : "")
                  << label << " (max abs err: " << max_abs_error
                  << ", max rel err: " << max_rel_error << ")" << std::endl;
        return true;
    } else {
        std::cerr << "[gradcheck] FAIL" << (label[0] ? "-" : "")
                  << label << std::endl;
        std::cerr << "  max abs error: " << max_abs_error
                  << " at (" << max_i << "," << max_j << ",ch=" << max_ch << ")" << std::endl;
        std::cerr << "  max rel error: " << max_rel_error << std::endl;
        std::cerr << "  analytical: " << analytical_grad.at(max_i, max_j, max_ch)
                  << "  numerical: " << numerical_grad.at(max_i, max_j, max_ch) << std::endl;
        return false;
    }
}

} // namespace CoreNNSpace

#endif // COREPP_AUTOGRAD_H
