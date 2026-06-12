#include <iostream>
#include "CorePP.h"
using namespace std;
using namespace CoreNNSpace;

int main() {
    cout << "===== Embedding Test =====" << endl;

    Embedding emb(10, 4);  // vocab=10, dim=4

    // Input: 3 indices
    Matrix<float> x(3, 1);
    x << 1 << 5 << 9;

    auto out = emb.forward(x);
    cout << "Shape: " << out.row << "x" << out.col << " (expect 3x4)" << endl;

    // Verify different indices → different embeddings
    bool diff = false;
    for (int e = 0; e < 4; ++e)
        if (std::abs(out.at(0,e) - out.at(1,e)) > 1e-6) diff = true;

    // Simple training: push embedding for index 0 toward [1,1,1,1]
    Matrix<float> target(3, 4);
    for (int e = 0; e < 4; ++e) target.at(0,e) = 1.0f;

    float mse0 = mse_loss(out, target);

    Optim optim({&emb}, SGD, 0.1f);
    for (int e = 0; e < 50; e++) {
        out = emb.forward(x);
        optim.step(out - target);
    }

    out = emb.forward(x);
    cout << "After training, emb[1][0]: " << out.at(0,0) << " (approaching 1.0)" << endl;
    cout << (mse_loss(out, target) < mse0 ? "Embedding Test PASSED" : "Embedding Test FAILED") << endl;
    return 0;
}
