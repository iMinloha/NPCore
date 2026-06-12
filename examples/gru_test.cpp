#include <iostream>
#include "CorePP.h"
using namespace std;
using namespace CoreNNSpace;

int main() {
    cout << "\n===== GRU Test: sequence memory =====" << endl;

    GRU gru(1, 6);
    Matrix<float> x(10, 1);
    x << 1 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0;

    Matrix<float> y(10, 6);
    y.at(9, 0) = 1; y.at(9, 1) = 1; y.at(9, 2) = 1;

    auto out = gru.forward(x);
    out.Analysis("Initial");
    cout << "MSE: " << mse_loss(out, y) << endl;

    Optim optim({&gru}, Adam, 0.005f);
    for (int e = 0; e < 600; e++) {
        out = gru.forward(x);
        optim.step(out - y);
        if (e % 200 == 0) printf("  epoch %3d: %.4f\n", e, mse_loss(out, y));
    }

    out = gru.forward(x);
    out.Analysis("Final");
    cout << "GRU Test COMPLETED" << endl;
    return 0;
}
