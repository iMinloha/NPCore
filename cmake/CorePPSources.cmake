# =================================[CorePP Source Files]================================

# --- Core module ---
set(COREPP_CORE_SOURCES CorePP/Core/RandomGenerator.cpp CorePP/DataLoader.cpp)
set(COREPP_CORE_HEADERS
    CorePP/Core/Assert.h CorePP/Core/ConvUtils.h CorePP/Core/CudaBridge.h
    CorePP/Core/GEMM.h CorePP/Core/LinearAlgebra.h CorePP/Core/Matrix.hpp
    CorePP/Core/Matrix.inl CorePP/Core/RandomGenerator.h CorePP/Core/Size.h
    CorePP/DataLoader.h
)

# --- Activations module ---
set(COREPP_ACTIVATIONS_SOURCES CorePP/Activations/Activation.cpp)
set(COREPP_ACTIVATIONS_HEADERS CorePP/Activations/Activation.h)

# --- Losses module ---
set(COREPP_LOSSES_HEADERS CorePP/Losses/Loss.h)

# --- Layers: Basic ---
set(COREPP_LAYERS_SOURCES
    CorePP/Layers/Module.cpp CorePP/Layers/Sequence.cpp
    CorePP/Layers/Basic/Linear.cpp CorePP/Layers/Basic/Flatten.cpp
)
set(COREPP_LAYERS_HEADERS
    CorePP/Layers/Module.h CorePP/Layers/ParamInit.h CorePP/Layers/Sequence.h
    CorePP/Layers/Basic/Linear.h CorePP/Layers/Basic/Flatten.h
    CorePP/Layers/Basic/Embedding.h CorePP/Layers/Basic/Dropout.h
)

# --- Layers: Conv ---
list(APPEND COREPP_LAYERS_SOURCES CorePP/Layers/Conv/Conv2d.cpp)
list(APPEND COREPP_LAYERS_HEADERS CorePP/Layers/Conv/Conv2d.h CorePP/Layers/Conv/Pool.h)

# --- Layers: Recurrent ---
list(APPEND COREPP_LAYERS_SOURCES
    CorePP/Layers/Recurrent/RNN.cpp CorePP/Layers/Recurrent/LSTM.cpp CorePP/Layers/Recurrent/GRU.cpp
)
list(APPEND COREPP_LAYERS_HEADERS
    CorePP/Layers/Recurrent/RNN.h CorePP/Layers/Recurrent/LSTM.h CorePP/Layers/Recurrent/GRU.h
)

# --- Layers: Normalization ---
list(APPEND COREPP_LAYERS_HEADERS
    CorePP/Layers/Normalization/BatchNorm.h CorePP/Layers/Normalization/LayerNorm.h
)

# --- Layers: Architecture ---
list(APPEND COREPP_LAYERS_HEADERS
    CorePP/Layers/Architecture/Residual.h CorePP/Layers/Architecture/ResNetBlock.h
)

# --- Optimizers ---
set(COREPP_OPTIMIZERS_SOURCES CorePP/Optimizers/Optimizer.cpp)
set(COREPP_OPTIMIZERS_HEADERS
    CorePP/Optimizers/Optimizer.h CorePP/Optimizers/SGD.h CorePP/Optimizers/Momentum.h
    CorePP/Optimizers/Adam.h CorePP/Optimizers/RMSProp.h CorePP/Optimizers/Adagrad.h
    CorePP/Optimizers/Adadelta.h CorePP/Optimizers/NAdam.h CorePP/Optimizers/RAdam.h
    CorePP/Optimizers/AdamW.h CorePP/Optimizers/LRScheduler.h
)

# --- Utils ---
set(COREPP_UTILS_SOURCES CorePP/Utils/Timer.cpp)
set(COREPP_UTILS_HEADERS CorePP/Utils/Timer.h)

# --- CUDA ---
set(COREPP_CUDA_HEADERS CorePP/Cuda/cuda_runtime.h)

# --- Module umbrellas ---
set(COREPP_MODULE_HEADERS
    CorePP/CorePP.h CorePP/Core.h CorePP/NN.h CorePP/Optim.h
    CorePP/Autograd.h CorePP/Model.h
)

# --- Aggregated ---
set(COREPP_ALL_SOURCES
    ${COREPP_CORE_SOURCES} ${COREPP_ACTIVATIONS_SOURCES} ${COREPP_LAYERS_SOURCES}
    ${COREPP_OPTIMIZERS_SOURCES} ${COREPP_UTILS_SOURCES}
)
set(COREPP_ALL_HEADERS
    ${COREPP_MODULE_HEADERS} ${COREPP_CORE_HEADERS} ${COREPP_ACTIVATIONS_HEADERS}
    ${COREPP_LOSSES_HEADERS} ${COREPP_LAYERS_HEADERS} ${COREPP_OPTIMIZERS_HEADERS}
    ${COREPP_UTILS_HEADERS} ${COREPP_CUDA_HEADERS}
)
