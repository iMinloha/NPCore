#include <iostream>
#include "CorePP.h"
using namespace std;
using namespace CoreNNSpace;

// Inline ResNet block: Conv(k3,s1,p1)→ReLU→Conv(k3,s1,p1) + skip → ReLU
class MyResBlock : public Module<float> {
    Conv2d *c1, *c2;
    Activation::ReLU *r1, *r2;
    Matrix<float>* skip = nullptr;
public:
    MyResBlock(int ch) {
        c1 = new Conv2d(ch, ch, 3, 1, 1);
        c2 = new Conv2d(ch, ch, 3, 1, 1);
        r1 = new Activation::ReLU();
        r2 = new Activation::ReLU();
    }
    ~MyResBlock() override { delete c1; delete c2; delete r1; delete r2; delete skip; }

    Matrix<float> forward(Matrix<float>& x) override {
        delete skip; skip = new Matrix<float>(x);
        auto o = c1->forward(x);
        o = r1->forward(o);
        o = c2->forward(o);
        int n = o.row * o.col * o.channel;
        for (int i = 0; i < n; ++i) o.data_ptr()[i] += skip->data_ptr()[i];
        o = r2->forward(o);
        auto* s = new Matrix<float>(o);
        output.push_back(s);
        return *s;
    }
    Matrix<float> backward(Matrix<float>& g) override {
        auto g0 = r2->backward(g);
        auto gs = Matrix<float>(g0);
        g0 = c2->backward(g0);
        g0 = r1->backward(g0);
        g0 = c1->backward(g0);
        int n = g0.row * g0.col * g0.channel;
        for (int i = 0; i < n; ++i) g0.data_ptr()[i] += gs.data_ptr()[i];
        return g0;
    }
    std::vector<Matrix<float>*> getParams() override {
        auto p = c1->getParams(); for (auto* m : c2->getParams()) p.push_back(m); return p;
    }
    std::vector<Matrix<float>*> getAllGrads() override {
        auto g = c1->getAllGrads(); for (auto* m : c2->getAllGrads()) g.push_back(m); return g;
    }
    Matrix<float>* getGard() override { return nullptr; }
    Matrix<float>* getOutput() override { return output.empty()?nullptr:output.back(); }
    void cuda() override { c1->cuda(); c2->cuda(); }
    void cpu()  override { c1->cpu();  c2->cpu();  }
    void eval()  { train_mode=false; c1->eval();  c2->eval();  }
    void train() { train_mode=true;  c1->train(); c2->train(); }
    void CleanGard() override {
        for(auto p:gard)delete p; gard.clear();
        for(auto p:output)delete p; output.clear();
        c1->CleanGard(); c2->CleanGard(); r1->CleanGard(); r2->CleanGard();
    }
};

int main() {
    cout << "\n===== ResNet Test: Conv+ResBlock+Pool+Linear =====" << endl;

    Sequence seq({
        new Conv2d(1, 4, 3, 1, 1),
        new Activation::ReLU(),
        new MyResBlock(4),
        new MaxPool2d(2, 2),
        new Flatten(),
        new Linear(36, 2),
    });

    Matrix<float> x(6, 6, 1);
    for (int i=0;i<6;++i) for (int j=0;j<6;++j) x.at(i,j,0)=(i*6+j+1)/36.0f;
    Matrix<float> y(2, 1); y << 1 << 0;

    auto out = seq.forward(x);
    out.Analysis("Initial");
    cout << "MSE: " << mse_loss(out, y) << endl;

    Optim opt(seq.getParams(), Adam, 0.005f);
    for (int e=0;e<300;e++) {
        out = seq.forward(x);
        opt.step(out - y);
        if (e%100==0) printf("  epoch %3d: %.6f\n", e, mse_loss(out,y));
    }

    out = seq.forward(x);
    out.Analysis("Final");
    y.Analysis("Target");
    cout << "ResNet Test COMPLETED" << endl;
    return 0;
}
