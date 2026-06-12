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

# --- Output all test binaries to tests/bin ---
set_target_properties(test_fnn test_cnn test_gradcheck test_rnn test_lstm
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
