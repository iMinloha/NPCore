# =================================[NPCore Source Files]================================

# --- Core module ---
set(NPCORE_CORE_SOURCES
    NPCore/Core/Matrix.cpp
    NPCore/Core/RandomGenerator.cpp
    NPCore/Core/LinearAlgebra.cpp
    NPCore/Core/Size.cpp
    NPCore/Core/GEMM.cpp
    NPCore/Core/ConvUtils.cpp
    NPCore/Core/CudaBridge.cpp
    NPCore/DataLoader.cpp
)
set(NPCORE_CORE_HEADERS
    NPCore/Core/Assert.h NPCore/Core/ConvUtils.h NPCore/Core/CudaBridge.h
    NPCore/Core/Export.h NPCore/Core/GEMM.h NPCore/Core/LinearAlgebra.h
    NPCore/Core/Matrix.hpp NPCore/Core/RandomGenerator.h NPCore/Core/Size.h
    NPCore/DataLoader.h
)

# --- Activations module ---
set(NPCORE_ACTIVATIONS_SOURCES NPCore/Activations/Activation.cpp)
set(NPCORE_ACTIVATIONS_HEADERS NPCore/Activations/Activation.h)

# --- Losses module ---
set(NPCORE_LOSSES_SOURCES NPCore/Losses/Loss.cpp)
set(NPCORE_LOSSES_HEADERS NPCore/Losses/Loss.h)

# --- Layers: Basic ---
set(NPCORE_LAYERS_SOURCES
    NPCore/Layers/Module.cpp NPCore/Layers/Sequence.cpp
    NPCore/Layers/ParamInit.cpp
    NPCore/Layers/Basic/Linear.cpp NPCore/Layers/Basic/Flatten.cpp
    NPCore/Layers/Basic/Embedding.cpp NPCore/Layers/Basic/Dropout.cpp
    NPCore/Layers/Basic/Concat.cpp
)
set(NPCORE_LAYERS_HEADERS
    NPCore/Layers/Module.h NPCore/Layers/ParamInit.h NPCore/Layers/Sequence.h
    NPCore/Layers/Basic/Linear.h NPCore/Layers/Basic/Flatten.h
    NPCore/Layers/Basic/Embedding.h NPCore/Layers/Basic/Dropout.h
    NPCore/Layers/Basic/Concat.h
)

# --- Layers: Conv ---
list(APPEND NPCORE_LAYERS_SOURCES
    NPCore/Layers/Conv/Conv1d.cpp
    NPCore/Layers/Conv/Conv2d.cpp NPCore/Layers/Conv/ConvTranspose2d.cpp
    NPCore/Layers/Conv/Pool.cpp
)
list(APPEND NPCORE_LAYERS_HEADERS
    NPCore/Layers/Conv/Conv1d.h
    NPCore/Layers/Conv/Conv2d.h NPCore/Layers/Conv/ConvTranspose2d.h NPCore/Layers/Conv/Pool.h
)

# --- Layers: Recurrent ---
list(APPEND NPCORE_LAYERS_SOURCES
    NPCore/Layers/Recurrent/RNN.cpp NPCore/Layers/Recurrent/LSTM.cpp NPCore/Layers/Recurrent/GRU.cpp
)
list(APPEND NPCORE_LAYERS_HEADERS
    NPCore/Layers/Recurrent/RNN.h NPCore/Layers/Recurrent/LSTM.h NPCore/Layers/Recurrent/GRU.h
)

# --- Layers: Normalization ---
list(APPEND NPCORE_LAYERS_SOURCES
    NPCore/Layers/Normalization/NormLayerBase.cpp
    NPCore/Layers/Normalization/BatchNorm.cpp NPCore/Layers/Normalization/LayerNorm.cpp
    NPCore/Layers/Normalization/GroupNorm.cpp
    NPCore/Layers/Normalization/InstanceNorm.cpp
)
list(APPEND NPCORE_LAYERS_HEADERS
    NPCore/Layers/Normalization/NormLayerBase.h
    NPCore/Layers/Normalization/BatchNorm.h NPCore/Layers/Normalization/LayerNorm.h
    NPCore/Layers/Normalization/GroupNorm.h
    NPCore/Layers/Normalization/InstanceNorm.h
)

# --- Layers: Attention ---
list(APPEND NPCORE_LAYERS_SOURCES
    NPCore/Layers/Attention/MultiHeadAttention.cpp
    NPCore/Layers/Attention/PositionalEncoding.cpp
    NPCore/Layers/Attention/TransformerEncoder.cpp
)
list(APPEND NPCORE_LAYERS_HEADERS
    NPCore/Layers/Attention/MultiHeadAttention.h
    NPCore/Layers/Attention/PositionalEncoding.h
    NPCore/Layers/Attention/TransformerEncoder.h
)

# --- Layers: Architecture ---
list(APPEND NPCORE_LAYERS_SOURCES
    NPCore/Layers/Architecture/Residual.cpp NPCore/Layers/Architecture/ResNetBlock.cpp
)
list(APPEND NPCORE_LAYERS_HEADERS
    NPCore/Layers/Architecture/Residual.h NPCore/Layers/Architecture/ResNetBlock.h
)

# --- Optimizers ---
set(NPCORE_OPTIMIZERS_SOURCES
    NPCore/Optimizers/Optimizer.cpp NPCore/Optimizers/OptimizerMethods.cpp
    NPCore/Optimizers/GradientClipping.cpp NPCore/Optimizers/LRScheduler.cpp
    NPCore/Optimizers/AdamW.cpp
)
set(NPCORE_OPTIMIZERS_HEADERS
    NPCore/Optimizers/Optimizer.h NPCore/Optimizers/SGD.h NPCore/Optimizers/Momentum.h
    NPCore/Optimizers/Adam.h NPCore/Optimizers/RMSProp.h NPCore/Optimizers/Adagrad.h
    NPCore/Optimizers/Adadelta.h NPCore/Optimizers/NAdam.h NPCore/Optimizers/RAdam.h
    NPCore/Optimizers/AdamW.h NPCore/Optimizers/LRScheduler.h
    NPCore/Optimizers/GradientClipping.h
)

# --- Utils ---
set(NPCORE_UTILS_SOURCES NPCore/Utils/Timer.cpp NPCore/Utils/Serializer.cpp NPCore/Utils/ONNXModel.cpp)
set(NPCORE_UTILS_HEADERS NPCore/Utils/Timer.h NPCore/Utils/Serializer.h NPCore/Utils/ONNXModel.h)

# --- CUDA ---
set(NPCORE_CUDA_HEADERS NPCore/Cuda/cuda_runtime.h)

# --- Module umbrellas ---
set(NPCORE_MODULE_SOURCES NPCore/Model.cpp NPCore/Autograd.cpp)
set(NPCORE_MODULE_HEADERS
    NPCore/NPCore.h NPCore/Core.h NPCore/NN.h NPCore/Optim.h
    NPCore/Autograd.h NPCore/Model.h
)

# --- Aggregated ---
set(NPCORE_ALL_SOURCES
    ${NPCORE_CORE_SOURCES} ${NPCORE_ACTIVATIONS_SOURCES} ${NPCORE_LOSSES_SOURCES}
    ${NPCORE_LAYERS_SOURCES} ${NPCORE_OPTIMIZERS_SOURCES} ${NPCORE_UTILS_SOURCES}
    ${NPCORE_MODULE_SOURCES}
)
set(NPCORE_ALL_HEADERS
    ${NPCORE_MODULE_HEADERS} ${NPCORE_CORE_HEADERS} ${NPCORE_ACTIVATIONS_HEADERS}
    ${NPCORE_LOSSES_HEADERS} ${NPCORE_LAYERS_HEADERS} ${NPCORE_OPTIMIZERS_HEADERS}
    ${NPCORE_UTILS_HEADERS} ${NPCORE_CUDA_HEADERS}
)
