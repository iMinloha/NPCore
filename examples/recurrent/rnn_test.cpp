#include <iostream>
#include "NPCore.h"
using namespace std;
using namespace NPCore;

int main() {
    cout << "\n===== RNN Test: sequence memory =====" << endl;

    RNN rnn(1, 6);
    Matrix<float> x(10, 1);
    x << 1 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0;

    Matrix<float> y(10, 6);
    y.at(9, 0) = 1; y.at(9, 1) = 1; y.at(9, 2) = 1;

    auto out = rnn.forward(x);
    out.Analysis("Initial");
    cout << "MSE: " << mse_loss(out, y) << endl;

    Optim optim(Adam_step, {&rnn}, 0.005f);
    for (int e = 0; e < 600; e++) {
        out = rnn.forward(x);
        optim.step(out - y);
        if (e % 200 == 0) printf("  epoch %3d: %.4f\n", e, mse_loss(out, y));
    }

    out = rnn.forward(x);
    out.Analysis("Final");
    cout << "RNN Test COMPLETED" << endl;
    return 0;
}
