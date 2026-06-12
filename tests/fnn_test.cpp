#include <iostream>
#include <chrono>
#include "CorePP.h"

using namespace std;
using namespace CoreNNSpace;

// =================================[FNN 全连接网络测试]================================
int main() {
    cout << "\n===== FNN Test: 4->8->16->8->4 =====" << endl;

    Sequence seq = Sequence({
        new Linear(4, 8),
        new Activation::Sigmoid(),
        new Linear(8, 16),
        new Activation::Sigmoid(),
        new Linear(16, 8),
        new Activation::Sigmoid(),
        new Linear(8, 4, XavierUniform),
    });

    Matrix<float> input = Matrix<float>(4, 1) << 3 << 4 << 2 << 1;
    Matrix<float> target = Matrix<float>(4, 1) << 1 << 0 << 0 << 0;

    Matrix<float> expect = seq.forward(input);
    cout << "Initial output:" << endl;
    expect.Analysis("PreView");
    cout << "Initial MSE: " << mse_loss(expect, target) << endl;

    float lr = 0.01f;
    int epoch = 200;
    Optim optim = Optim(seq.getParams(), RMSProp, lr);

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

    cout << "FNN Test COMPLETED" << endl;
    return 0;
}
