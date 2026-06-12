#include <iostream>
#include "CorePP.h"
using namespace std;
using namespace CoreNNSpace;

int main() {
    cout << "===== GRU Test =====" << endl;
    GRU gru(2, 4);

    // Sequence: 5 steps, 2 features
    Matrix<float> x(5, 2);
    float vals[5][2] = {{0.1,0.2},{0.3,0.1},{0.5,0.4},{0.2,0.3},{0.1,0.5}};
    for (int t = 0; t < 5; ++t) for (int f = 0; f < 2; ++f) x.at(t,f) = vals[t][f];

    Matrix<float> y(5, 4); y << 0.3 << 0.5 << 0.2 << 0.1 << 0;

    float mse0 = mse_loss(gru.forward(x), y);
    cout << "Initial MSE: " << mse0 << endl;

    Optim optim({&gru}, Adam, 0.005f);
    for (int e = 0; e < 500; e++) {
        auto out = gru.forward(x);
        optim.step(out - y);
        if (e % 200 == 0) cout << "  epoch " << e << ": " << mse_loss(out, y) << endl;
    }

    auto out = gru.forward(x);
    printf("Final: [%.3f, %.3f, %.3f, %.3f, %.3f]  MSE: %.4f\n",
           out.at(0,0), out.at(1,0), out.at(2,0), out.at(3,0), out.at(4,0),
           mse_loss(out, y));
    cout << (mse_loss(out, y) < mse0 ? "GRU Test PASSED" : "GRU Test FAILED") << endl;
    return 0;
}
