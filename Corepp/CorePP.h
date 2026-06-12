#ifndef COREPP_COREPP_H
#define COREPP_COREPP_H

// =================================[CorePP — C++ Deep Learning Library]================================
//
// Configuration Macros:
//   AlgorithmAccelerator  - Enable block-tiled matrix multiplication (L1 cache optimized)
//   __AVX__               - Auto-detected by CMake: enables AVX SIMD instructions
//   __ARM_NEON            - Auto-detected by CMake: enables ARM NEON SIMD instructions
//   COREPP_HAS_OPENMP     - Auto-detected by CMake: enables OpenMP parallelism
//   COREPP_BUILD_DLL      - Defined when building CorePP as DLL
//   COREPP_USE_DLL        - Defined when linking against CorePP DLL

// --- DLL export/import ---
#if defined(COREPP_BUILD_DLL)
    #define COREPP_API __declspec(dllexport)
#elif defined(COREPP_USE_DLL)
    #define COREPP_API __declspec(dllimport)
#else
    #define COREPP_API
#endif

// Enable algorithm acceleration by default
#define AlgorithmAccelerator

// --- Foundation Types (always included) ---
#include "Core.h"

// --- Neural Network Layers ---
#include "NN.h"

// --- Optimizers ---
#include "Optim.h"

// --- Loss Functions ---
#include "Losses/Loss.h"

// --- High-Level API ---
#include "Model.h"

// --- Utilities ---
#include "Utils/Timer.h"

#endif // COREPP_COREPP_H
