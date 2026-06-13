// =================================[CorePP Example: Fully Connected Network]================================
// 4 inputs -> 8 -> 16 -> 8 -> 4 outputs, Sigmoid activation, MSE loss, RMSProp optimizer
#include <iostream>
#include <chrono>
#include "CorePP.h"
using namespace std;
using namespace CoreNNSpace;

int main() {
    cout << "\n===== FNN: 4->8->16->8->4 =====" << endl;

    // 1. 构建网络
    Sequence seq({
        new Linear(4, 8),   new Activation::Sigmoid(),  // 第1层
        new Linear(8, 16),  new Activation::Sigmoid(),  // 第2层
        new Linear(16, 8),  new Activation::Sigmoid(),  // 第3层
        new Linear(8, 4, XavierUniform),                 // 输出层
    });

    // 2. 准备数据
    Matrix<float> x(4, 1);  x << 3 << 4 << 2 << 1;       // 输入 (列向量)
    Matrix<float> y(4, 1);  y << 1 << 0 << 0 << 0;       // 目标 one-hot

    // 3. 初始预测
    auto out = seq.forward(x);
    out.Analysis("Initial");
    cout << "MSE: " << mse_loss(out, y) << endl;

    // 4. 训练
    Optim optim(seq.getParams(), RMSProp, 0.01f);
    auto t0 = chrono::steady_clock::now();
    for (int e = 0; e < 300; e++) {
        out = seq.forward(x);
        optim.step(mse_loss_grad(out, y));                // 梯度 = out - target
        if (e % 100 == 0) cout << "  epoch " << e << ": " << mse_loss(out, y) << endl;
    }
    auto dur = chrono::duration<double,milli>(chrono::steady_clock::now()-t0).count();

    // 5. 结果
    out = seq.forward(x);
    out.Analysis("Final");
    y.Analysis("Target");
    cout << "Time: " << dur << "ms" << endl;
    return 0;
}
