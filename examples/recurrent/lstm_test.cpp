#include <iostream>
#include "NPCore.h"
using namespace std;
using namespace NPCore;

int main() {
    cout << "\n===== LSTM Test: long-term memory =====" << endl;

    LSTM lstm(1, 4);
    Matrix<float> x(10, 1);
    x << 1 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0;

    Matrix<float> y(10, 4);
    y.at(9, 0) = 1; y.at(9, 1) = 1; y.at(9, 2) = 1; y.at(9, 3) = 1;

    auto out = lstm.forward(x);
    out.Analysis("Initial");
    cout << "MSE: " << mse_loss(out, y) << endl;

    Optim optim({&lstm}, Adam, 0.01f);
    for (int e = 0; e < 800; e++) {
        out = lstm.forward(x);
        optim.step(out - y);
        if (e % 200 == 0) printf("  epoch %3d: %.4f\n", e, mse_loss(out, y));
    }

    out = lstm.forward(x);
    out.Analysis("Final");
    cout << "LSTM Test COMPLETED" << endl;
    return 0;
}
