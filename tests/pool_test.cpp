#include <iostream>
#include "CorePP.h"
using namespace std;
using namespace CoreNNSpace;

int main() {
    cout << "===== MaxPool2d Test =====" << endl;

    // 4x4x2 input → 2x2x2 output (pool=2, stride=2)
    Matrix<float> x(4, 4, 2);
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                x.at(i,j,c) = (float)((i*4+j+1) * (c+1));

    MaxPool2d pool(2, 2);
    auto out = pool.forward(x);
    out.Analysis("MaxPool 2x2");

    // Verify correct shape and values
    bool ok = (out.row == 2 && out.col == 2 && out.channel == 2);
    // Check known max values: 4x4 input = [[1,2,3,4],[5,6,7,8],...]
    // Block(0,0): max of [[1,2],[5,6]] = 6 (ch0), 12 (ch1)
    // Block(0,1): max of [[3,4],[7,8]] = 8 (ch0), 16 (ch1)
    if (ok && out.at(0,0,0) != 6)  ok = false;
    if (ok && out.at(0,1,0) != 8)  ok = false;
    cout << "Shape: " << out.row << "x" << out.col << "x" << out.channel << endl;
    cout << (ok ? "MaxPool2d Test PASSED" : "MaxPool2d Test FAILED") << endl;
    return 0;
}
