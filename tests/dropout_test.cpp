#include <iostream>
#include "CorePP.h"
using namespace std;
using namespace CoreNNSpace;

int main() {
    cout << "===== Dropout Test =====" << endl;

    // Create a matrix of ones
    Matrix<float> x(10, 10);
    for (int i = 0; i < 100; ++i) x.data_ptr()[i] = 1.0f;

    Dropout dropout(0.5f);  // 50% dropout rate

    // Train mode: roughly half should be zero, half should be 2.0 (scale = 1/(1-0.5) = 2)
    auto out = dropout.forward(x);
    int zeros = 0, twos = 0;
    for (int i = 0; i < 100; ++i) {
        if (out.data_ptr()[i] == 0.0f) zeros++;
        else if (out.data_ptr()[i] > 1.9f && out.data_ptr()[i] < 2.1f) twos++;
    }

    cout << "Train mode: zeros=" << zeros << " scaled=" << twos
         << " (expect ~50 each)" << endl;

    // Eval mode: all should be 1.0
    dropout.eval();
    out = dropout.forward(x);
    int ones = 0;
    for (int i = 0; i < 100; ++i)
        if (out.data_ptr()[i] > 0.99f && out.data_ptr()[i] < 1.01f) ones++;

    cout << "Eval mode: ones=" << ones << " (expect 100)" << endl;
    cout << (ones == 100 ? "Dropout Test PASSED" : "Dropout Test FAILED") << endl;
    return 0;
}
