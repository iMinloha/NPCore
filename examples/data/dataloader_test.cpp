// =================================[NPCore Example: DataLoader]================================
// Demonstrates InMemoryLoader train/test split + Trainer.fit(DataLoader&)
#include <iostream>
#include "NPCore.h"
#include "Model.h"
using namespace std;
using namespace NPCore;

// ---- 自定义 DataLoader 示例 ----
class MyLoader : public DataLoader {
    vector<Matrix<float>> inputs_, targets_;
    int cursor_ = 0;
public:
    void add(float x1, float x2, float y0) {
        Matrix<float> xi(2,1); xi << x1 << x2;
        Matrix<float> yi(1,1); yi << y0;
        inputs_.push_back(xi); targets_.push_back(yi);
    }
    int  num_samples() const override { return (int)inputs_.size(); }
    void reset() override { cursor_ = 0; }
    bool next_batch(Matrix<float>& x, Matrix<float>& y) override {
        if (cursor_ >= (int)inputs_.size()) { cursor_ = 0; return false; }
        x = inputs_[cursor_]; y = targets_[cursor_]; cursor_++; return true;
    }
};

int main() {
    cout << "\n===== DataLoader Example =====" << endl;

    // 1. 准备数据 (用 InMemoryLoader)
    InMemoryLoader loader;
    for (int i = 0; i < 100; i++) {
        float a = RandomGenerator::uniform<float>(-1, 1);
        float b = RandomGenerator::uniform<float>(-1, 1);
        Matrix<float> x(2, 1); x << a << b;
        Matrix<float> y(1, 1); y << (a + b > 0 ? 1.0f : 0.0f);  // 二分类
        loader.add_sample(x, y);
    }
    loader.split(0.7f, 0.3f);  // 70% train, 30% test
    loader.train();              // 切换到训练集
    cout << "Train: " << loader.num_samples() << " samples" << endl;

    // 2. 网络
    auto net = nn::FNN({2, 8, 1}, nn::Sigmoid);
    net.train();

    // 3. 用 DataLoader 训练
    loader.train();
    cout << "Training samples: " << loader.num_samples() << endl;

    Optim optim(Adam_step, net.modules(), 0.01f);
    Matrix<float> x, y;
    for (int e = 0; e < 50; e++) {
        loader.reset();
        float total_loss = 0; int count = 0;
        while (loader.next_batch(x, y)) {
            auto out = net.forward(x);
            optim.step(loss_grad(out, y, nn::MSE));
            total_loss += loss_val(out, y, nn::MSE);
            count++;
        }
        if (e % 10 == 0) printf("  epoch %2d: loss=%.4f\n", e, total_loss / count);
    }

    // 4. 测试
    loader.test();
    net.eval();
    int correct = 0, total = 0;
    while (loader.next_batch(x, y)) {
        auto out = net.forward(x);
        if ((out.at(0,0) > 0.5f) == (y.at(0,0) > 0.5f)) correct++;
        total++;
    }
    printf("Test accuracy: %d/%d = %.1f%%\n", correct, total, 100.0f*correct/total);

    // 5. 自定义 DataLoader
    MyLoader my;
    my.add(0.2f, 0.8f, 1.0f);
    my.add(-0.5f, -0.3f, 0.0f);
    my.add(0.6f, 0.1f, 1.0f);
    cout << "Custom loader: " << my.num_samples() << " samples" << endl;

    cout << "DataLoader Example COMPLETED" << endl;
    return 0;
}
