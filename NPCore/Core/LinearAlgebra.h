#ifndef NPCORE_CORE_LINEARALGEBRA_H
#define NPCORE_CORE_LINEARALGEBRA_H

#include "Matrix.hpp"

// =================================[Gram-Schmidt 正交化]================================
// 对矩阵列向量进行 Gram-Schmidt 正交归一化

template<typename T>
void GramSchmidtOrthogonalization(Matrix<T>& mat);

#endif // NPCORE_CORE_LINEARALGEBRA_H
