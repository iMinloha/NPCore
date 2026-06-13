#include <iostream>
#include <chrono>
#include "CorePP.h"
using namespace std;
using namespace CoreNNSpace;

int main() {
    cout << "\n===== CNN Test: Conv+Pool+Flatten+Linear =====" << endl;

    Sequence seq({
        new Conv2d(1, 4, 3, 1, 0),       // 8x8x1 -> 6x6x4
        new Activation::ReLU(),
        new MaxPool2d(2, 2),               // 6x6x4 -> 3x3x4
        new Conv2d(4, 4, 3, 1, 0),        // 3x3x4 -> 1x1x4
        new Activation::ReLU(),
        new Flatten(),                      // -> 4
        new Linear(4, 2),
    });

    Matrix<float> x(8, 8, 1);
    InitMatrixFunc(x, Uniform, {.a = -0.5, .b = 0.5});
    Matrix<float> y(2, 1); y << 1 << 0;

    auto out = seq.forward(x);
    out.Analysis("Initial");
    cout << "MSE: " << mse_loss(out, y) << endl;

    auto t0 = chrono::steady_clock::now();
    Optim optim(seq.getParams(), Adam, 0.002f);
    for (int e = 0; e < 300; e++) {
        out = seq.forward(x);
        optim.step(mse_loss_grad(out, y));
        if (e % 100 == 0) printf("  epoch %3d: %.6f\n", e, mse_loss(out, y));
    }
    auto dur = chrono::duration<double,milli>(chrono::steady_clock::now()-t0).count();

    out = seq.forward(x);
    out.Analysis("Final");
    y.Analysis("Target");
    printf("Time: %.1fms\n", dur);
    cout << "CNN Test COMPLETED" << endl;
    return 0;
}
