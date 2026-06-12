#include <iostream>
#include "CorePP.h"

using namespace std;
using namespace CoreNNSpace;

// LSTM: memorize a pattern across a long sequence
int main() {
    cout << "\n===== LSTM Test: Long-term Memory =====" << endl;

    const int seq_len = 10, input_size = 1, hidden_size = 4;

    // Input: [1,0,0,0,0,0,0,0,0,0] — remember first value
    Matrix<float> input(seq_len, input_size);
    input.at(0, 0) = 1.0f;
    for (int t = 1; t < seq_len; ++t) input.at(t, 0) = 0.0f;

    // Target: output 1 at last time step (long-term memory test)
    // LSTM output is (seq_len, hidden_size) — supervise all dims
    Matrix<float> target(seq_len, hidden_size);
    for (int i = 0; i < hidden_size; ++i)
        target.at(seq_len - 1, i) = 1.0f;

    LSTM lstm(input_size, hidden_size);
    cout << "LSTM created: hidden=" << hidden_size << endl;

    Matrix<float> out = lstm.forward(input);
    cout << "Initial last output dim0: " << out.at(seq_len-1, 0)
         << " (target: 1.0)" << endl;

    // Train with Adam
    Optim optim({&lstm}, Adam, 0.01f);
    for (int epoch = 0; epoch < 1000; epoch++) {
        out = lstm.forward(input);
        Matrix<float> loss = out - target;  // (seq_len, hidden_size)
        optim.step(loss);

        if (epoch % 50 == 0) {
            float mse = 0;
            for (int t = 0; t < seq_len; ++t)
                mse += loss.at(t, 0) * loss.at(t, 0);
            cout << "Epoch " << epoch << "  MSE: " << mse / seq_len << endl;
        }
    }

    out = lstm.forward(input);
    cout << "Final outputs dim0: [";
    for (int t = 0; t < seq_len; ++t)
        cout << out.at(t, 0) << (t < seq_len-1 ? ", " : "");
    cout << "]" << endl;
    cout << "Last output dim0: " << out.at(seq_len-1, 0) << " (target: 1.0)" << endl;

    float final_mse = 0;
    for (int t = 0; t < seq_len; ++t) {
        float diff = out.at(t, 0) - target.at(t, 0);
        final_mse += diff * diff;
    }
    cout << "Final MSE: " << final_mse / seq_len << endl;
    cout << "LSTM Test COMPLETED" << endl;
    return 0;
}
