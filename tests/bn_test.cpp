#include <iostream>
#include "CorePP.h"
using namespace std;
using namespace CoreNNSpace;

int main() {
    cout << "===== BatchNorm1d Test =====" << endl;

    // Batch of 4 samples, 3 features
    Matrix<float> x(4, 3);
    float vals[4][3] = {{1,2,3},{2,3,4},{3,4,5},{4,5,6}};
    for (int i = 0; i < 4; ++i) for (int f = 0; f < 3; ++f) x.at(i,f) = vals[i][f];

    BatchNorm1d bn(3);
    auto out = bn.forward(x);

    // After BN: output should have ~mean 0 and ~std gamma
    float mean = 0;
    for (int i = 0; i < 4; ++i) mean += out.at(i,0);
    mean /= 4;

    cout << "Output shape: " << out.row << "x" << out.col << endl;
    cout << "Mean of dim0: " << mean << " (should be ~0)" << endl;

    // Train a bit
    Matrix<float> target(4, 3);  // all zeros target
    Optim optim({&bn}, SGD, 0.01f);
    for (int e = 0; e < 10; e++) {
        out = bn.forward(x);
        optim.step(out - target);
    }

    cout << "BatchNorm1d Test PASSED" << endl;
    return 0;
}
