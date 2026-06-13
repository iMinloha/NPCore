#ifndef NPCORE_OPTIMIZERS_GRADIENTCLIPPING_H
#define NPCORE_OPTIMIZERS_GRADIENTCLIPPING_H

#include <vector>
#include "Core/Matrix.hpp"

namespace NPCore {

template<typename T> class Module;

// =================================[Gradient Clipping]================================
// Reference: Pascanu et al. (2013)
// Prevents exploding gradients by constraining gradient magnitude.

struct GradientClipping {
    static void clip_by_norm(const std::vector<Module<float>*>& modules, float max_norm);
    static float clip_by_norm(const std::vector<Module<float>*>& modules,
                              float max_norm, bool);
    static void clip_by_value(const std::vector<Module<float>*>& modules,
                              float min_val, float max_val);
    static float clip_by_norm(Matrix<float>& grad, float max_norm);
    static void clip_by_value(Matrix<float>& grad, float min_val, float max_val);
};

} // namespace NPCore
#endif
