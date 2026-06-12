# =================================[CorePP Tests]================================

set(TEST_LIBS CorePP::CorePP)
set(TEST_DIR  ${CMAKE_BINARY_DIR}/tests)

# --- Gradient verification ---
add_executable(test_gradcheck gradcheck_test.cpp)
target_link_libraries(test_gradcheck PRIVATE ${TEST_LIBS})

# --- Feedforward ---
add_executable(test_fnn fnn_test.cpp)
target_link_libraries(test_fnn PRIVATE ${TEST_LIBS})

# --- Convolutional ---
add_executable(test_cnn cnn_test.cpp)
target_link_libraries(test_cnn PRIVATE ${TEST_LIBS})

add_executable(test_resnet resnet_test.cpp)
target_link_libraries(test_resnet PRIVATE ${TEST_LIBS})

# --- Recurrent ---
add_executable(test_rnn rnn_test.cpp)
target_link_libraries(test_rnn PRIVATE ${TEST_LIBS})

add_executable(test_lstm lstm_test.cpp)
target_link_libraries(test_lstm PRIVATE ${TEST_LIBS})

add_executable(test_gru gru_test.cpp)
target_link_libraries(test_gru PRIVATE ${TEST_LIBS})

# --- Output ---
set(ALL_TESTS
    test_gradcheck test_fnn test_cnn test_resnet
    test_rnn test_lstm test_gru
)
set_target_properties(${ALL_TESTS} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_DIR})
