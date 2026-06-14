#ifndef NPCORE_AUTOGRAD_H
#define NPCORE_AUTOGRAD_H

// =================================[NPCore/Autograd - Gradient Checking Utility]================================
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

#include <functional>
#include "Core/Matrix.hpp"

namespace NPCore {

// =================================[Numerical Gradient (Finite Differences)]================================
// Computes dL/dx at point x using central finite differences.
// f: loss function that takes a Matrix and returns a scalar
// x: point at which to compute gradient
// epsilon: perturbation size (default 1e-5)
template<typename T>
Matrix<T> numerical_gradient(std::function<T(const Matrix<T>&)> f,
                              const Matrix<T>& x, T epsilon = T(1e-5));

// =================================[Gradient Check]================================
// Compares analytical gradient against numerical gradient.
// Returns true if the relative error is below threshold for all elements.
// Prints detailed comparison to stderr.
template<typename T>
bool gradcheck(const Matrix<T>& analytical_grad,
               const Matrix<T>& numerical_grad,
               T threshold = T(1e-3),
               const char* label = "");

} // namespace NPCore

#endif // NPCORE_AUTOGRAD_H
