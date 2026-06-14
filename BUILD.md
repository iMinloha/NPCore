# NPCore — Build Guide

## Prerequisites

| Dependency | Version | Notes |
|-----------|---------|-------|
| CMake | ≥ 3.18 | |
| C++ Compiler | GCC 11+ / MinGW-w64 / MSVC 2022+ | C++20 required |
| OpenMP | — | Optional but recommended (auto-detected) |
| CUDA Toolkit | 11.x / 12.x | Optional; `-DNPCORE_ENABLE_CUDA=ON` |

### Windows (MinGW-w64)

```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-openmp
```

### Windows (MSVC)

Install Visual Studio 2022 with the "Desktop development with C++" workload.

## Quick Build (CPU only)

```bash
cd D:\C++\Korea_C++
mkdir build && cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build . -j8
```

## Build with CUDA

```bash
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DNPCORE_ENABLE_CUDA=ON
cmake --build . -j8
```

## Build as Shared Library (DLL)

```bash
cmake .. -G "MinGW Makefiles" -DBUILD_SHARED_LIBS=ON
cmake --build . -j8
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_SHARED_LIBS` | `OFF` | Build `.dll`/`.so` instead of `.a`/`.lib` |
| `NPCORE_ENABLE_CUDA` | `OFF` | Enable CUDA GPU acceleration |
| `BUILD_EXAMPLES` | `ON` | Build example/test executables |

## Output Structure

```
build/
├── lib/libNPCore.a          # Static library
├── bin/                      # Example executables
└── install/                  # After cmake --install .
    ├── include/NPCore/       # All headers
    └── lib/libNPCore.a       # Library
```

## Source File List

### Static Library (`libNPCore`)

```
# === Core ===
NPCore/Core/RandomGenerator.cpp
NPCore/Core/LinearAlgebra.cpp
NPCore/Core/Size.cpp
NPCore/Core/GEMM.cpp
NPCore/Core/ConvUtils.cpp
NPCore/Core/CudaBridge.cpp
NPCore/DataLoader.cpp

# === Autograd ===
NPCore/Autograd.cpp

# === Losses ===
NPCore/Losses/Loss.cpp

# === Activations ===
NPCore/Activations/Activation.cpp

# === Layers ===
NPCore/Layers/Module.cpp
NPCore/Layers/Sequence.cpp
NPCore/Layers/ParamInit.cpp
NPCore/Layers/Basic/Linear.cpp
NPCore/Layers/Basic/Flatten.cpp
NPCore/Layers/Basic/Embedding.cpp
NPCore/Layers/Basic/Dropout.cpp
NPCore/Layers/Conv/Conv2d.cpp
NPCore/Layers/Conv/ConvTranspose2d.cpp
NPCore/Layers/Conv/Pool.cpp
NPCore/Layers/Recurrent/RNN.cpp
NPCore/Layers/Recurrent/LSTM.cpp
NPCore/Layers/Recurrent/GRU.cpp
NPCore/Layers/Normalization/BatchNorm.cpp
NPCore/Layers/Normalization/LayerNorm.cpp
NPCore/Layers/Normalization/GroupNorm.cpp
NPCore/Layers/Attention/MultiHeadAttention.cpp
NPCore/Layers/Architecture/Residual.cpp
NPCore/Layers/Architecture/ResNetBlock.cpp

# === Optimizers ===
NPCore/Optimizers/Optimizer.cpp
NPCore/Optimizers/OptimizerMethods.cpp
NPCore/Optimizers/GradientClipping.cpp
NPCore/Optimizers/LRScheduler.cpp
NPCore/Optimizers/AdamW.cpp

# === Model API ===
NPCore/Model.cpp

# === Utils ===
NPCore/Utils/Timer.cpp
```

### CUDA (optional, when `-DNPCORE_ENABLE_CUDA=ON`)

```
NPCore/Cuda/cuda_device.cu
NPCore/Cuda/cuda_gemm.cu
NPCore/Cuda/cuda_elementwise.cu
NPCore/Cuda/cuda_rnn.cu
```

### All Headers (reference)

```
NPCore/NPCore.h        NPCore/Core.h         NPCore/NN.h
NPCore/Optim.h         NPCore/Autograd.h     NPCore/Model.h
NPCore/DataLoader.h

NPCore/Core/Assert.h         NPCore/Core/ConvUtils.h
NPCore/Core/CudaBridge.h     NPCore/Core/GEMM.h
NPCore/Core/LinearAlgebra.h  NPCore/Core/Matrix.hpp
NPCore/Core/Matrix.inl       NPCore/Core/RandomGenerator.h
NPCore/Core/Size.h

NPCore/Activations/Activation.h
NPCore/Losses/Loss.h

NPCore/Layers/Module.h       NPCore/Layers/ParamInit.h
NPCore/Layers/Sequence.h
NPCore/Layers/Basic/Linear.h   NPCore/Layers/Basic/Flatten.h
NPCore/Layers/Basic/Embedding.h NPCore/Layers/Basic/Dropout.h
NPCore/Layers/Conv/Conv2d.h    NPCore/Layers/Conv/ConvTranspose2d.h
NPCore/Layers/Conv/Pool.h
NPCore/Layers/Recurrent/RNN.h  NPCore/Layers/Recurrent/LSTM.h
NPCore/Layers/Recurrent/GRU.h
NPCore/Layers/Normalization/BatchNorm.h  NPCore/Layers/Normalization/LayerNorm.h
NPCore/Layers/Normalization/GroupNorm.h
NPCore/Layers/Attention/MultiHeadAttention.h
NPCore/Layers/Architecture/Residual.h     NPCore/Layers/Architecture/ResNetBlock.h

NPCore/Optimizers/Optimizer.h  NPCore/Optimizers/SGD.h
NPCore/Optimizers/Momentum.h   NPCore/Optimizers/Adam.h
NPCore/Optimizers/RMSProp.h    NPCore/Optimizers/Adagrad.h
NPCore/Optimizers/Adadelta.h   NPCore/Optimizers/NAdam.h
NPCore/Optimizers/RAdam.h      NPCore/Optimizers/AdamW.h
NPCore/Optimizers/LRScheduler.h NPCore/Optimizers/GradientClipping.h

NPCore/Utils/Timer.h
```

## Header-Only Principle

This project follows a strict **zero-logic-in-headers** rule:

- All function/method implementations live in `.cpp` files
- Template functions use **explicit instantiation** (`template class Foo<float>;`)
- The only exception is `Core/Assert.h` (preprocessor macro — must be in a header)

### Matrix Template

`Matrix<T>` uses the `.inl` pattern: declarations in `Matrix.hpp`, implementations in `Matrix.inl` (included by the header). This is the only template-heavy type — all other templates use explicit instantiation.

## Compiler Flags

| Flag | Purpose |
|------|---------|
| `-mavx2 -mfma` | AVX2 + FMA for GEMM micro-kernel |
| `-march=native` | Native CPU tuning (MinGW) |
| `-fopenmp` | OpenMP parallelism |
| `/arch:AVX2` | AVX2 (MSVC) |
| `/MP` | Multi-processor compilation (MSVC) |

## Example: Manual Compilation (Single File)

```bash
g++ -std=c++20 -mavx2 -mfma -march=native -fopenmp -O2 \
    -I NPCore \
    your_file.cpp \
    -o your_program
```

## Example: Linking Against Installed NPCore

```bash
g++ -std=c++20 -O2 \
    -I install/include/NPCore \
    your_demo.cpp \
    install/lib/libNPCore.a \
    -fopenmp \
    -o demo
```
