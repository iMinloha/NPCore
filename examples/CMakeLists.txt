set(LIBS NPCore::NPCore)
set(DIR  ${CMAKE_BINARY_DIR}/examples)

# ============================================================
# Categories:
#   unit/         — unit tests & gradient checks
#   basic/        — basic layers (Linear, etc.)
#   conv/         — convolution & pooling
#   recurrent/    — RNN, LSTM, GRU
#   normalization/— BatchNorm, LayerNorm, GroupNorm, InstanceNorm
#   attention/    — MultiHeadAttention, Transformer, PositionalEncoding
#   optimizers/   — optimizers & gradient clipping
#   architecture/ — high-level architectures (ResNet, etc.)
#   data/         — data loading
#   serialize/    — model save/load
#   onnx/         — ONNX export
# ============================================================

# --- Unit ---
add_executable(test_gradcheck unit/gradcheck_test.cpp)
add_executable(test_use_bias  unit/test_use_bias.cpp)
add_executable(test_trainer   unit/test_trainer.cpp)
add_executable(test_losses    unit/test_losses.cpp)
target_link_libraries(test_gradcheck PRIVATE ${LIBS})
target_link_libraries(test_use_bias  PRIVATE ${LIBS})
target_link_libraries(test_trainer   PRIVATE ${LIBS})
target_link_libraries(test_losses    PRIVATE ${LIBS})

# --- Basic ---
add_executable(test_fnn basic/fnn_test.cpp)
target_link_libraries(test_fnn PRIVATE ${LIBS})

# --- Convolution ---
add_executable(test_cnn              conv/cnn_test.cpp)
add_executable(test_conv1d           conv/test_conv1d.cpp)
add_executable(test_convtranspose2d  conv/test_convtranspose2d.cpp)
add_executable(test_pooling          conv/test_pooling.cpp)
target_link_libraries(test_cnn             PRIVATE ${LIBS})
target_link_libraries(test_conv1d         PRIVATE ${LIBS})
target_link_libraries(test_convtranspose2d PRIVATE ${LIBS})
target_link_libraries(test_pooling         PRIVATE ${LIBS})

# --- Recurrent ---
add_executable(test_rnn  recurrent/rnn_test.cpp)
add_executable(test_lstm recurrent/lstm_test.cpp)
add_executable(test_gru  recurrent/gru_test.cpp)
target_link_libraries(test_rnn  PRIVATE ${LIBS})
target_link_libraries(test_lstm PRIVATE ${LIBS})
target_link_libraries(test_gru  PRIVATE ${LIBS})

# --- Normalization ---
add_executable(test_batchnorm2d   normalization/test_batchnorm2d.cpp)
add_executable(test_groupnorm     normalization/test_groupnorm.cpp)
add_executable(test_instancenorm  normalization/test_instancenorm.cpp)
target_link_libraries(test_batchnorm2d   PRIVATE ${LIBS})
target_link_libraries(test_groupnorm     PRIVATE ${LIBS})
target_link_libraries(test_instancenorm  PRIVATE ${LIBS})

# --- Attention ---
add_executable(test_mha         attention/test_mha.cpp)
add_executable(test_transformer attention/test_transformer.cpp)
target_link_libraries(test_mha         PRIVATE ${LIBS})
target_link_libraries(test_transformer PRIVATE ${LIBS})

# --- Optimizers ---
add_executable(test_gradclip optimizers/test_gradclip.cpp)
target_link_libraries(test_gradclip PRIVATE ${LIBS})

# --- Architecture ---
add_executable(test_resnet architecture/resnet_test.cpp)
target_link_libraries(test_resnet PRIVATE ${LIBS})

# --- Data ---
add_executable(test_dataloader data/dataloader_test.cpp)
target_link_libraries(test_dataloader PRIVATE ${LIBS})

# --- Serialization ---
add_executable(test_serialize serialize/test_serialize.cpp)
target_link_libraries(test_serialize PRIVATE ${LIBS})

# --- ONNX ---
add_executable(test_onnx        onnx/test_onnx.cpp)
add_executable(test_onnx_import onnx/test_onnx_import.cpp)
target_link_libraries(test_onnx        PRIVATE ${LIBS})
target_link_libraries(test_onnx_import PRIVATE ${LIBS})

# --- All targets ---
set(ALL_EXAMPLES
    # Unit
    test_gradcheck test_use_bias test_trainer test_losses
    # Basic
    test_fnn
    # Convolution
    test_cnn test_conv1d test_convtranspose2d test_pooling
    # Recurrent
    test_rnn test_lstm test_gru
    # Normalization
    test_batchnorm2d test_groupnorm test_instancenorm
    # Attention
    test_mha test_transformer
    # Optimizers
    test_gradclip
    # Architecture
    test_resnet
    # Data
    test_dataloader
    # Serialization
    test_serialize
    # ONNX
    test_onnx test_onnx_import
)
set_target_properties(${ALL_EXAMPLES} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${DIR})
