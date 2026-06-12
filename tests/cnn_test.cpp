#include <iostream>
#include <chrono>
#include "CorePP.h"

using namespace std;
using namespace CoreNNSpace;

// =================================[CNN 卷积网络测试]================================
int main() {
    cout << "\n===== CNN Test: Conv2d->Sigmoid->Conv2d->Flatten->Linear =====" << endl;

    Sequence seq = Sequence({
        new Conv2d(3, 4, 3, 1, 0),    // (8,8,3) -> (6,6,4)
        new Activation::Sigmoid(),
        new Conv2d(4, 2, 3, 1, 0),    // (6,6,4) -> (4,4,2)
        new Activation::Sigmoid(),
        new Flatten(),                  // (4,4,2) -> (32,1)
        new Linear(32, 4),             // (32,1) -> (4,1)
    });

    Matrix<float> input(8, 8, 3);
    InitMatrixFunc(input, InitMode::Uniform, {.a = -1, .b = 1});

    Matrix<float> target = Matrix<float>(4, 1) << 1 << 0 << 0 << 0;

    Matrix<float> expect = seq.forward(input);
    cout << "Initial output:" << endl;
    expect.Analysis("PreView");
    cout << "Initial MSE: " << mse_loss(expect, target) << endl;

    float lr = 0.002f;
    int epoch = 500;
    Optim optim = Optim(seq.getParams(), Adam, lr);

    auto beforeTime = chrono::steady_clock::now();

    for (int i = 0; i < epoch; i++) {
        expect = seq.forward(input);
        Matrix<float> loss = mse_loss_grad(expect, target);
        optim.step(loss);

        if (i % 50 == 0) {
            cout << "Epoch " << i << "  MSE: " << mse_loss(expect, target) << endl;
        }
    }

    auto afterTime = chrono::steady_clock::now();
    double duration_ms = chrono::duration<double, milli>(afterTime - beforeTime).count();
    cout << "Training time: " << duration_ms << "ms" << endl;

    expect = seq.forward(input);
    cout << "Final output:" << endl;
    expect.Analysis("Final");
    cout << "Target: [1, 0, 0, 0]" << endl;

    cout << "CNN Test COMPLETED" << endl;
    return 0;
}
