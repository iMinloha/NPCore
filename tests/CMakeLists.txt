# =================================[CorePP Tests]================================

# --- FNN test ---
add_executable(test_fnn
    fnn_test.cpp
)
target_link_libraries(test_fnn PRIVATE CorePP::CorePP)

# --- CNN test ---
add_executable(test_cnn
    cnn_test.cpp
)
target_link_libraries(test_cnn PRIVATE CorePP::CorePP)

# --- Gradient check test ---
add_executable(test_gradcheck
    gradcheck_test.cpp
)
target_link_libraries(test_gradcheck PRIVATE CorePP::CorePP)

# --- RNN test ---
add_executable(test_rnn
    rnn_test.cpp
)
target_link_libraries(test_rnn PRIVATE CorePP::CorePP)

# --- LSTM test ---
add_executable(test_lstm
    lstm_test.cpp
)
target_link_libraries(test_lstm PRIVATE CorePP::CorePP)

# --- GRU test ---
add_executable(test_gru gru_test.cpp)
target_link_libraries(test_gru PRIVATE CorePP::CorePP)

# --- MaxPool2d test ---
add_executable(test_pool pool_test.cpp)
target_link_libraries(test_pool PRIVATE CorePP::CorePP)

# --- BatchNorm test ---
add_executable(test_bn bn_test.cpp)
target_link_libraries(test_bn PRIVATE CorePP::CorePP)

# --- Dropout test ---
add_executable(test_dropout dropout_test.cpp)
target_link_libraries(test_dropout PRIVATE CorePP::CorePP)

# --- Embedding test ---
add_executable(test_embed embed_test.cpp)
target_link_libraries(test_embed PRIVATE CorePP::CorePP)

# --- Residual test ---
add_executable(test_residual residual_test.cpp)
target_link_libraries(test_residual PRIVATE CorePP::CorePP)

# --- Output all test binaries ---
set_target_properties(
    test_fnn test_cnn test_gradcheck test_rnn test_lstm
    test_gru test_pool test_bn test_dropout test_embed test_residual
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
