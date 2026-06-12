# =================================[CorePP Source Files]================================
# Explicit source lists — no GLOB_RECURSE so new files are noticed immediately

# --- Core module ---
set(COREPP_CORE_SOURCES
    CorePP/Core/RandomGenerator.cpp
)

set(COREPP_CORE_HEADERS
    CorePP/Core/Assert.h
    CorePP/Core/ConvUtils.h
    CorePP/Core/CudaBridge.h
    CorePP/Core/LinearAlgebra.h
    CorePP/Core/Matrix.hpp
    CorePP/Core/Matrix.inl
    CorePP/Core/RandomGenerator.h
    CorePP/Core/Size.h
)

# --- Layers module ---
set(COREPP_LAYERS_SOURCES
    CorePP/Layers/Activation.cpp
    CorePP/Layers/Conv2d.cpp
    CorePP/Layers/Flatten.cpp
    CorePP/Layers/GRU.cpp
    CorePP/Layers/Linear.cpp
    CorePP/Layers/LSTM.cpp
    CorePP/Layers/Module.cpp
    CorePP/Layers/RNN.cpp
    CorePP/Layers/Sequence.cpp
)

set(COREPP_LAYERS_HEADERS
    CorePP/Layers/Activation.h
    CorePP/Layers/BatchNorm.h
    CorePP/Layers/Conv2d.h
    CorePP/Layers/Dropout.h
    CorePP/Layers/Embedding.h
    CorePP/Layers/Flatten.h
    CorePP/Layers/GRU.h
    CorePP/Layers/Linear.h
    CorePP/Layers/LSTM.h
    CorePP/Layers/Module.h
    CorePP/Layers/ParamInit.h
    CorePP/Layers/Pool.h
    CorePP/Layers/Residual.h
    CorePP/Layers/RNN.h
    CorePP/Layers/Sequence.h
)

# --- Optimizers module ---
set(COREPP_OPTIMIZERS_SOURCES
    CorePP/Optimizers/Optimizer.cpp
)

set(COREPP_OPTIMIZERS_HEADERS
    CorePP/Optimizers/Optimizer.h
)

# --- Utils module ---
set(COREPP_UTILS_SOURCES
    CorePP/Utils/Timer.cpp
)

set(COREPP_UTILS_HEADERS
    CorePP/Utils/Timer.h
)

# --- CUDA module ---
set(COREPP_CUDA_HEADERS
    CorePP/Cuda/cuda_runtime.h
)

# --- Module umbrella headers (top-level includes) ---
set(COREPP_MODULE_HEADERS
    CorePP/CorePP.h
    CorePP/Core.h
    CorePP/NN.h
    CorePP/Optim.h
    CorePP/Loss.h
    CorePP/Autograd.h
    CorePP/Model.h
)

# --- Aggregated lists ---
set(COREPP_ALL_SOURCES
    ${COREPP_CORE_SOURCES}
    ${COREPP_LAYERS_SOURCES}
    ${COREPP_OPTIMIZERS_SOURCES}
    ${COREPP_UTILS_SOURCES}
)

set(COREPP_ALL_HEADERS
    ${COREPP_MODULE_HEADERS}
    ${COREPP_CORE_HEADERS}
    ${COREPP_LAYERS_HEADERS}
    ${COREPP_OPTIMIZERS_HEADERS}
    ${COREPP_UTILS_HEADERS}
    ${COREPP_CUDA_HEADERS}
)
