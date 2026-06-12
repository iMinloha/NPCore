set(LIBS CorePP::CorePP)
set(DIR  ${CMAKE_BINARY_DIR}/examples)

add_executable(example_gradcheck gradcheck_test.cpp)
target_link_libraries(example_gradcheck PRIVATE ${LIBS})

add_executable(example_fnn fnn_test.cpp)
target_link_libraries(example_fnn PRIVATE ${LIBS})

add_executable(example_cnn cnn_test.cpp)
target_link_libraries(example_cnn PRIVATE ${LIBS})

add_executable(example_rnn rnn_test.cpp)
target_link_libraries(example_rnn PRIVATE ${LIBS})

add_executable(example_lstm lstm_test.cpp)
target_link_libraries(example_lstm PRIVATE ${LIBS})

add_executable(example_gru gru_test.cpp)
target_link_libraries(example_gru PRIVATE ${LIBS})

add_executable(example_resnet resnet_test.cpp)
target_link_libraries(example_resnet PRIVATE ${LIBS})

add_executable(example_dataloader dataloader_test.cpp)
target_link_libraries(example_dataloader PRIVATE ${LIBS})

set(ALL_EXAMPLES
    example_gradcheck example_fnn example_cnn
    example_rnn example_lstm example_gru example_resnet
    example_dataloader
)
set_target_properties(${ALL_EXAMPLES} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${DIR})
