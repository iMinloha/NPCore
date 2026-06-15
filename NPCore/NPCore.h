#ifndef NPCORE_CORE_H
#define NPCORE_CORE_H

// =================================[NPCore - C++ Deep Learning Library]================================
//
// Configuration Macros:
//   AlgorithmAccelerator  - Enable block-tiled matrix multiplication (L1 cache optimized)
//   __AVX__               - Auto-detected by CMake: enables AVX SIMD instructions
//   __ARM_NEON            - Auto-detected by CMake: enables ARM NEON SIMD instructions
//   NPCORE_HAS_OPENMP     - Auto-detected by CMake: enables OpenMP parallelism
//   NPCORE_BUILD_DLL      - Defined when building NPCore as DLL
//   NPCORE_USE_DLL        - Defined when linking against NPCore DLL

// --- DLL export/import ---
#include "Core/Export.h"

// Enable algorithm acceleration by default
#define AlgorithmAccelerator

// --- Foundation Types (always included) ---
#include "Core.h"

// --- Autograd ---
#include "Autograd.h"

// --- Neural Network Layers ---
#include "NN.h"

// --- Optimizers ---
#include "Optim.h"

// --- Loss Functions ---
#include "Losses/Loss.h"

// --- Data Loading ---
#include "DataLoader.h"

// --- High-Level API ---
#include "Model.h"

// --- Utilities ---
#include "Utils/Timer.h"
#include "Utils/Serializer.h"
#include "Utils/ONNXModel.h"

#endif // NPCORE_CORE_H
