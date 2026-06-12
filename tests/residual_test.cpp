#include <iostream>
#include "CorePP.h"
using namespace std;
using namespace CoreNNSpace;

int main() {
    cout << "===== Residual Test =====" << endl;

    // Residual wrapper around Linear(4,4) + Sigmoid
    // y = sigmoid(W*x + b) + x    (skip connection)

    auto* sub = new Linear(4, 4);
    Residual res(sub);

    Matrix<float> x(4, 1); x << 1 << 2 << 3 << 4;
    Matrix<float> y(4, 1); y << 1 << 0 << 0 << 0;

    auto out = res.forward(x);
    cout << "Shape: " << out.row << "x" << out.col << " (expect 4x1)" << endl;

    // Verify: residual output ≠ linear-only output (skip connection adds x)
    auto lin_out = sub->forward(x);
    bool different = false;
    for (int i = 0; i < 4; ++i)
        if (std::abs(out.at(i,0) - lin_out.at(i,0)) > 1e-6) different = true;
    cout << "Skip connection active: " << (different ? "YES" : "NO") << endl;

    // Train
    float mse0 = mse_loss(out, y);
    Optim optim({&res}, SGD, 0.01f);
    for (int e = 0; e < 50; e++) {
        out = res.forward(x);
        optim.step(out - y);
    }
    cout << "MSE: " << mse0 << " → " << mse_loss(out, y) << endl;
    cout << (mse_loss(out, y) < mse0 ? "Residual Test PASSED" : "Residual Test FAILED") << endl;
    return 0;
}
