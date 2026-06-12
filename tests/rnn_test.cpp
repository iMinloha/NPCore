#include <iostream>
#include "CorePP.h"

using namespace std;
using namespace CoreNNSpace;

// RNN sequence prediction: given [a,b,c], predict [b,c,d] (shift by 1)
int main() {
    cout << "\n===== RNN Test: Sequence Prediction =====" << endl;

    const int seq_len = 5, input_size = 2, hidden_size = 4;

    // Input sequence: 5 time steps, each with 2 features
    Matrix<float> input(seq_len, input_size);
    float vals[5][2] = {{0.1f,0.2f},{0.3f,0.1f},{0.5f,0.4f},{0.2f,0.3f},{0.1f,0.5f}};
    for (int t = 0; t < seq_len; ++t)
        for (int f = 0; f < input_size; ++f)
            input.at(t, f) = vals[t][f];

    // Target: predict next value of feature 0
    // RNN output is (seq_len, hidden_size) — we supervise only first dim
    Matrix<float> target(seq_len, hidden_size);
    float tgt[5] = {0.3f, 0.5f, 0.2f, 0.1f, 0.0f};
    for (int t = 0; t < seq_len; ++t)
        target.at(t, 0) = tgt[t];

    RNN rnn(input_size, hidden_size);
    cout << "RNN created: hidden=" << hidden_size << endl;

    Matrix<float> out = rnn.forward(input);
    cout << "Initial output dim0: [" << out.at(0,0) << ", " << out.at(1,0)
         << ", " << out.at(2,0) << ", " << out.at(3,0) << "]" << endl;

    // Train with Adam
    Optim optim({&rnn}, Adam, 0.005f);
    for (int epoch = 0; epoch < 800; epoch++) {
        out = rnn.forward(input);
        Matrix<float> loss = out - target;  // (seq_len, hidden_size)
        optim.step(loss);

        if (epoch % 200 == 0) {
            float mse = 0;
            for (int t = 0; t < seq_len; ++t)
                mse += loss.at(t, 0) * loss.at(t, 0);  // only track dim 0
            cout << "Epoch " << epoch << "  MSE: " << mse / seq_len << endl;
        }
    }

    out = rnn.forward(input);
    cout << "Final output dim0: [" << out.at(0,0) << ", " << out.at(1,0)
         << ", " << out.at(2,0) << ", " << out.at(3,0) << ", " << out.at(4,0) << "]" << endl;
    cout << "Target:           [" << tgt[0] << ", " << tgt[1] << ", " << tgt[2]
         << ", " << tgt[3] << ", " << tgt[4] << "]" << endl;

    float final_mse = 0;
    for (int t = 0; t < seq_len; ++t) {
        float diff = out.at(t, 0) - tgt[t];
        final_mse += diff * diff;
    }
    cout << "Final MSE: " << final_mse / seq_len << endl;
    cout << "RNN Test COMPLETED" << endl;
    return 0;
}
