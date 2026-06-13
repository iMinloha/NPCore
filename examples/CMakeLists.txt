set(LIBS NPCore::NPCore)
set(DIR  ${CMAKE_BINARY_DIR}/examples)

# ============================================================
# Categories:
#   unit/         — unit tests & gradient checks
#   basic/        — basic layers (Linear, etc.)
#   conv/         — convolution & pooling
#   recurrent/    — RNN, LSTM, GRU
#   normalization/— BatchNorm, LayerNorm, GroupNorm
#   attention/    — MultiHeadAttention (Transformer)
#   optimizers/   — optimizers & gradient clipping
#   architecture/ — high-level architectures (ResNet, etc.)
#   data/         — data loading
# ============================================================

# --- Unit ---
add_executable(test_gradcheck unit/gradcheck_test.cpp)
target_link_libraries(test_gradcheck PRIVATE ${LIBS})

# --- Basic ---
add_executable(test_fnn basic/fnn_test.cpp)
target_link_libraries(test_fnn PRIVATE ${LIBS})

# --- Convolution ---
add_executable(test_cnn              conv/cnn_test.cpp)
add_executable(test_convtranspose2d  conv/test_convtranspose2d.cpp)
add_executable(test_pooling          conv/test_pooling.cpp)
target_link_libraries(test_cnn             PRIVATE ${LIBS})
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
add_executable(test_batchnorm2d normalization/test_batchnorm2d.cpp)
add_executable(test_groupnorm   normalization/test_groupnorm.cpp)
target_link_libraries(test_batchnorm2d PRIVATE ${LIBS})
target_link_libraries(test_groupnorm   PRIVATE ${LIBS})

# --- Attention ---
add_executable(test_mha attention/test_mha.cpp)
target_link_libraries(test_mha PRIVATE ${LIBS})

# --- Optimizers ---
add_executable(test_gradclip optimizers/test_gradclip.cpp)
target_link_libraries(test_gradclip PRIVATE ${LIBS})

# --- Architecture ---
add_executable(test_resnet architecture/resnet_test.cpp)
target_link_libraries(test_resnet PRIVATE ${LIBS})

# --- Data ---
add_executable(test_dataloader data/dataloader_test.cpp)
target_link_libraries(test_dataloader PRIVATE ${LIBS})

# --- All targets ---
set(ALL_EXAMPLES
    # Unit
    test_gradcheck
    # Basic
    test_fnn
    # Convolution
    test_cnn test_convtranspose2d test_pooling
    # Recurrent
    test_rnn test_lstm test_gru
    # Normalization
    test_batchnorm2d test_groupnorm
    # Attention
    test_mha
    # Optimizers
    test_gradclip
    # Architecture
    test_resnet
    # Data
    test_dataloader
)
set_target_properties(${ALL_EXAMPLES} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${DIR})
