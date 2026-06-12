#include <iostream>
#include <chrono>
#include "CorePP.h"

using namespace std;
using namespace CoreNNSpace;

int main() {
    cout << "\n===== FNN Test: 4->8->16->8->4 =====" << endl;

    Sequence seq = Sequence({
        new Linear(4, 8),       new Activation::Sigmoid(),
        new Linear(8, 16),      new Activation::Sigmoid(),
        new Linear(16, 8),      new Activation::Sigmoid(),
        new Linear(8, 4, XavierUniform),
    });

    Matrix<float> input(4, 1);  input << 3 << 4 << 2 << 1;
    Matrix<float> target(4, 1); target << 1 << 0 << 0 << 0;

    auto out = seq.forward(input);
    out.Analysis("Initial");
    cout << "MSE: " << mse_loss(out, target) << endl;

    Optim optim(seq.getParams(), RMSProp, 0.01f);
    auto t0 = chrono::steady_clock::now();
    for (int e = 0; e < 300; e++) {
        out = seq.forward(input);
        optim.step(mse_loss_grad(out, target));
        if (e % 100 == 0) cout << "  epoch " << e << ": " << mse_loss(out, target) << endl;
    }
    auto dur = chrono::duration<double,milli>(chrono::steady_clock::now()-t0).count();

    out = seq.forward(input);
    out.Analysis("Final");
    target.Analysis("Target");
    cout << "Time: " << dur << "ms" << endl;
    cout << "FNN Test COMPLETED" << endl;
    return 0;
}
