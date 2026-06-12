// =================================[CorePP DLL 动态链接示例]================================
// 编译: g++ -std=c++20 -I../../CorePP main.cpp -L../../_build/lib -lCorePP -fopenmp
// 运行: 确保 CorePP.dll 在 PATH 或同目录

#include <iostream>
#include "CorePP.h"

using namespace std;
using namespace CoreNNSpace;

int main() {
    cout << "CorePP Dynamic Linking Example\n" << endl;

    // ===== FNN: 全连接网络 =====
    {
        cout << "--- FNN: 4->8->4 ---" << endl;
        auto net = nn::FNN({4, 8, 4}, nn::Sigmoid);

        Matrix<float> x(4, 1); x << 0.5 << -0.3 << 0.8 << -0.1;
        Matrix<float> y(4, 1); y << 1 << 0 << 0 << 0;

        auto out = net.forward(x);
        cout << "Before training: [" << out.at(0,0) << ", " << out.at(1,0)
             << ", " << out.at(2,0) << ", " << out.at(3,0) << "]" << endl;

        nn::Trainer(net, nn::MSE, Optim(net.getParams(), Adam, 0.01f))
            .fit(x, y, 300, [](int e, float loss) {
                if (e % 100 == 0) printf("  epoch %3d: loss=%.6f\n", e, loss);
            });

        out = net.forward(x);
        out.Analysis("FNN Result");
    }

    // ===== CNN: 卷积网络 =====
    {
        cout << "\n--- CNN: Conv->Sigmoid->Flatten->Linear ---" << endl;
        auto net = nn::CNN({3, 4, 4}, 3, nn::Sigmoid, 4);

        Matrix<float> x(8, 8, 3);
        InitMatrixFunc(x, Uniform, {.a = -0.5, .b = 0.5});
        Matrix<float> y(4, 1); y << 1 << 0 << 0 << 0;

        nn::Trainer(net, nn::MSE, Optim(net.getParams(), Adam, 0.002f))
            .fit(x, y, 400, [](int e, float loss) {
                if (e % 100 == 0) printf("  epoch %3d: loss=%.6f\n", e, loss);
            });

        auto out = net.forward(x);
        out.Analysis("CNN Result");
    }

    // ===== RNN: 循环网络 =====
    {
        cout << "\n--- RNN: sequence ---" << endl;
        RNN rnn(2, 4);

        Matrix<float> x(5, 2);
        float vals[5][2] = {{0.1,0.2},{0.3,0.1},{0.5,0.4},{0.2,0.3},{0.1,0.5}};
        for (int t = 0; t < 5; ++t)
            for (int f = 0; f < 2; ++f) x.at(t, f) = vals[t][f];

        Matrix<float> y(5, 4);
        y << 0.3 << 0.5 << 0.2 << 0.1 << 0;

        nn::Trainer(rnn, nn::MSE, Optim({&rnn}, Adam, 0.005f))
            .fit(x, y, 600, [](int e, float loss) {
                if (e % 200 == 0) printf("  epoch %3d: loss=%.6f\n", e, loss);
            });

        auto out = rnn.forward(x);
        printf("  output: [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
               out.at(0,0), out.at(1,0), out.at(2,0), out.at(3,0), out.at(4,0));
    }

    // ===== LSTM: 长短期记忆 =====
    {
        cout << "\n--- LSTM: long-term memory ---" << endl;
        LSTM lstm(1, 4);

        Matrix<float> x(10, 1);
        x << 1 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0;
        Matrix<float> y(10, 4);
        for (int i = 0; i < 4; ++i) y.at(9, i) = 1;

        nn::Trainer(lstm, nn::MSE, Optim({&lstm}, Adam, 0.01f))
            .fit(x, y, 800, [](int e, float loss) {
                if (e % 200 == 0) printf("  epoch %3d: loss=%.6f\n", e, loss);
            });

        auto out = lstm.forward(x);
        printf("  last output: %.4f (target 1.0)\n", out.at(9, 0));
    }

    cout << "\nAll examples completed." << endl;
    return 0;
}
