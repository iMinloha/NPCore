// =================================[NPCore Full Test Suite]================================
// Tests: use_bias, Conv1d, TransformerEncoder, InstanceNorm, save/load, ONNX, FNN+dropout
//
// Build: g++ -std=c++20 -I NPCore test_all.cpp -L build/lib -lNPCore -fopenmp -o test_all

#include "NPCore.h"
#include "Model.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cstring>
#include <fstream>

using namespace NPCore;
using namespace NPCore::nn;

static int passed = 0, failed = 0;
#define TEST(name) do { printf("  %-50s ... ", name); } while(0)
#define OK() do { printf("PASS\n"); passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); failed++; } while(0)
#define CHECK(cond, msg) do { if(!(cond)){FAIL(msg);return;} } while(0)
#define CLOSE(a,b,eps,msg) do { if(fabsf((a)-(b))>(eps)){printf("FAIL: %s (%.6f vs %.6f)\n",msg,(float)(a),(float)(b));failed++;return;} } while(0)

// =================================[1. use_bias parameter]================================
static void test_use_bias() {
    TEST("Linear use_bias=false");
    {
        Linear lin(4, 8, InitMode::Uniform, false);  // no bias
        auto p = lin.getParams();
        CHECK(p.size() == 1, "expected 1 param (weight only)");

        Matrix<float> x(4,1); x << 1 << 2 << 3 << 4;
        auto out = lin.forward(x);
        CHECK(out.row == 8 && out.col == 1, "output shape");
        lin.CleanGard();
    }
    OK();

    TEST("Linear use_bias=true (default)");
    {
        Linear lin(4, 8);  // default: bias=true
        CHECK(lin.getParams().size() == 2, "expected 2 params");
        Matrix<float> x(4,1); x << 1 << 2 << 3 << 4;
        auto out = lin.forward(x);
        CHECK(out.row == 8 && out.col == 1, "output shape");
        lin.CleanGard();
    }
    OK();

    TEST("Conv2d use_bias=false");
    {
        Conv2d conv(3, 16, 3, 1, 0, InitMode::XavierUniform, false);
        CHECK(conv.getParams().size() == 1, "expected 1 param");
        Matrix<float> img(32, 32, 3);
        for(int i=0;i<32*32*3;++i) img.data_ptr()[i]=0.1f;
        auto out = conv.forward(img);
        CHECK(out.row == 30 && out.col == 30 && out.channel == 16, "output shape");
        conv.CleanGard();
    }
    OK();

    TEST("Conv2d use_bias=true (default)");
    {
        Conv2d conv(3, 16, 3, 1, 0);
        CHECK(conv.getParams().size() == 2, "expected 2 params");
        Matrix<float> img(32, 32, 3);
        for(int i=0;i<32*32*3;++i) img.data_ptr()[i]=0.1f;
        auto out = conv.forward(img);
        CHECK(out.row == 30 && out.col == 30 && out.channel == 16, "output shape");
        conv.CleanGard();
    }
    OK();
}

// =================================[2. Conv1d]================================
static void test_conv1d() {
    TEST("Conv1d forward shape");
    {
        Conv1d conv(3, 8, 3, 1, 0);
        Matrix<float> seq(10, 3); // 10 time steps, 3 features
        for(int i=0;i<30;++i) seq.data_ptr()[i]=0.1f;
        auto out = conv.forward(seq);
        CHECK(out.row == 8 && out.col == 8, "output shape (L-K+1=8, out_ch=8)");
        conv.CleanGard();
    }
    OK();

    TEST("Conv1d forward+backward grad check");
    {
        Conv1d conv(2, 4, 3, 1, 0);
        conv.train();
        Matrix<float> x(6, 2);
        for(int i=0;i<12;++i) x.data_ptr()[i] = (float)(i+1)*0.1f;
        auto out = conv.forward(x);
        Matrix<float> grad(out.row, out.col);
        for(int i=0;i<grad.row*grad.col;++i) grad.data_ptr()[i]=0.01f;
        auto dx = conv.backward(grad);
        CHECK(dx.row == 6 && dx.col == 2, "gradient shape matches input");
        // Verify weight gradient exists
        CHECK(conv.getWeightGrad() != nullptr, "weight grad computed");
        CHECK(conv.getWeightGrad()->row > 0, "weight grad has rows");
        conv.CleanGard();
    }
    OK();
}

// =================================[3. TransformerEncoder]================================
static void test_transformer() {
    TEST("TransformerEncoderLayer forward");
    {
        TransformerEncoderLayer enc(16, 2, 64);
        Matrix<float> x(5, 16); // 5 tokens, d_model=16
        for(int i=0;i<80;++i) x.data_ptr()[i]=0.1f;
        auto out = enc.forward(x);
        CHECK(out.row == 5 && out.col == 16, "output shape matches input");
        enc.CleanGard();
    }
    OK();

    TEST("TransformerEncoderLayer backward");
    {
        TransformerEncoderLayer enc(16, 2, 64);
        enc.train();
        Matrix<float> x(5, 16);
        for(int i=0;i<80;++i) x.data_ptr()[i]=0.1f;
        auto out = enc.forward(x);
        Matrix<float> grad(5, 16);
        for(int i=0;i<80;++i) grad.data_ptr()[i]=0.01f;
        auto dx = enc.backward(grad);
        CHECK(dx.row == 5 && dx.col == 16, "gradient shape");
        CHECK(enc.getAllGrads().size() > 0, "gradients computed");
        enc.CleanGard();
    }
    OK();

    TEST("TransformerEncoder stack");
    {
        TransformerEncoder enc(32, 4, 128, 2);
        Matrix<float> x(10, 32);
        for(int i=0;i<320;++i) x.data_ptr()[i]=0.1f;
        auto out = enc.forward(x);
        CHECK(out.row == 10 && out.col == 32, "stack output shape");
        enc.CleanGard();
    }
    OK();

    TEST("TransformerEncoder + PositionalEncoding");
    {
        std::vector<Module<float>*> layers = {
            new PositionalEncoding(16),
            new TransformerEncoderLayer(16, 2, 64)
        };
        Sequence model(layers);
        Matrix<float> x(5, 16);
        for(int i=0;i<80;++i) x.data_ptr()[i]=0.1f;
        auto out = model.forward(x);
        CHECK(out.row == 5 && out.col == 16, "PE+TF output shape");
    }
    OK();
}

// =================================[4. InstanceNorm]================================
static void test_instancenorm() {
    TEST("InstanceNorm1d forward");
    {
        InstanceNorm1d inorm(8);
        Matrix<float> x(4, 8);
        for(int i=0;i<32;++i) x.data_ptr()[i]=(float)(i+1);
        auto out = inorm.forward(x);
        CHECK(out.row == 4 && out.col == 8, "output shape");
        inorm.CleanGard();
    }
    OK();

    TEST("InstanceNorm1d backward");
    {
        InstanceNorm1d inorm(8);
        inorm.train();
        Matrix<float> x(4, 8);
        for(int i=0;i<32;++i) x.data_ptr()[i]=(float)(i+1);
        auto out = inorm.forward(x);
        Matrix<float> grad(4, 8);
        for(int i=0;i<32;++i) grad.data_ptr()[i]=0.1f;
        auto dx = inorm.backward(grad);
        CHECK(dx.row == 4 && dx.col == 8, "grad shape");
        inorm.CleanGard();
    }
    OK();

    TEST("InstanceNorm2d forward");
    {
        InstanceNorm2d in2d(3);
        Matrix<float> img(8, 8, 3);
        for(int i=0;i<192;++i) img.data_ptr()[i]=(float)(i+1)*0.01f;
        auto out = in2d.forward(img);
        CHECK(out.row == 8 && out.col == 8 && out.channel == 3, "output shape");
        in2d.CleanGard();
    }
    OK();
}

// =================================[5. Serialization]================================
static void test_serialization() {
    TEST("save_model + load_model_weights");
    {
        std::vector<Module<float>*> layers = {
            new Linear(4, 8),
            new Activation::ReLU(),
            new Linear(8, 2)
        };
        Sequence model(layers);

        // Save
        bool ok = save_model(model, "test_model.np");
        CHECK(ok, "save_model returned true");

        // Create identical architecture and load weights
        std::vector<Module<float>*> layers2 = {
            new Linear(4, 8),
            new Activation::ReLU(),
            new Linear(8, 2)
        };
        Sequence model2(layers2);

        ok = load_model_weights(model2, "test_model.np");
        CHECK(ok, "load_model_weights returned true");

        // Verify weights match (compare Linear layers)
        auto p1 = model.getLayers()[0]->getParams();
        auto p2 = model2.getLayers()[0]->getParams();
        CHECK(p1.size() == p2.size(), "same number of params");
        if(p1.size() == p2.size() && p1.size() >= 2) {
            for(int i=0;i<p1[0]->row*p1[0]->col;++i)
                CLOSE(p1[0]->data_ptr()[i], p2[0]->data_ptr()[i], 0.001f, "weight mismatch");
        }
        std::remove("test_model.np");
    }
    OK();

    TEST("save_sequence + load_sequence roundtrip");
    {
        std::vector<Module<float>*> layers = {
            new Linear(3, 6),
            new Activation::Tanh()
        };
        Sequence src(layers);

        std::vector<LayerTag> tags = {LayerTag::Linear, LayerTag::Unknown};
        bool ok = save_sequence(src, "test_seq.np", tags);
        CHECK(ok, "save_sequence ok");

        auto loaded = load_sequence("test_seq.np");
        CHECK(loaded.getLayers().size() >= 1, "loaded has layers");

        auto sp = src.getLayers()[0]->getParams();
        auto lp = loaded.getLayers()[0]->getParams();
        if(sp.size()>=2 && lp.size()>=2) {
            for(int i=0;i<sp[0]->row*sp[0]->col;++i)
                CLOSE(sp[0]->data_ptr()[i], lp[0]->data_ptr()[i], 0.001f, "weight mismatch after load");
        }
        std::remove("test_seq.np");
    }
    OK();
}

// =================================[6. ONNX Export]================================
static void test_onnx() {
    TEST("ONNX export simple MLP");
    {
        std::vector<Module<float>*> layers = {
            new Linear(2, 8),
            new Activation::ReLU(),
            new Linear(8, 2),
            new Activation::SoftMax()
        };
        Sequence model(layers);

        bool ok = export_onnx(model, {1, 2}, "test_model.onnx", "TestMLP");
        CHECK(ok, "export_onnx returned true");

        // Check file exists and has content
        std::ifstream f("test_model.onnx", std::ios::binary);
        CHECK(f.good(), "file exists");
        f.seekg(0, std::ios::end);
        CHECK(f.tellg() > 50, "file has reasonable size");
        f.close();
        std::remove("test_model.onnx");
    }
    OK();

    TEST("ONNX export CNN");
    {
        std::vector<Module<float>*> layers = {
            new Conv2d(1, 8, 3, 1, 0),
            new Activation::ReLU(),
            new MaxPool2d(2, 2),
            new NPCore::Flatten(),
            new Linear(13*13*8, 10)  // for 28x28 input
        };
        Sequence model(layers);

        bool ok = export_onnx(model, {1, 1, 28, 28}, "test_cnn.onnx", "TestCNN");
        CHECK(ok, "export_onnx CNN ok");
        std::ifstream f("test_cnn.onnx", std::ios::binary);
        f.seekg(0, std::ios::end);
        CHECK(f.good() && f.tellg() > 0, "CNN onnx file exists");
        f.close();
        std::remove("test_cnn.onnx");
    }
    OK();
}

// =================================[7. Trainer with new features]================================
static void test_trainer() {
    TEST("Trainer with Sequence and Tanh");
    {
        auto net = nn::FNN({2, 4, 2}, nn::Tanh);
        Matrix<float> x(2,1); x << 1 << 2;
        Matrix<float> y(2,1); y << 1 << 0;
        nn::Trainer trainer(net, nn::MSE, Adam(0.01f));
        trainer.fit(x, y, 2, nullptr);
        CHECK(true, "trainer ran without crash");
    }
    OK();

    TEST("Trainer with CrossEntropy loss");
    {
        auto net = nn::FNN({3, 8, 3}, nn::ReLU);
        Matrix<float> x(3,1); x << 1 << 2 << 3;
        Matrix<float> y(3,1); y << 1 << 0 << 0;
        nn::Trainer trainer(net, nn::CrossEntropy, Adam(0.01f));
        trainer.fit(x, y, 2, nullptr);
        CHECK(true, "CE trainer ran");
    }
    OK();

    TEST("Trainer bind changes optimizer");
    {
        auto net = nn::FNN({2, 4, 1}, nn::Sigmoid);
        Matrix<float> x(2,1); x << 1 << 2;
        Matrix<float> y(1,1); y << 0.5f;
        nn::Trainer trainer(net, nn::MSE, SGD(0.1f));
        trainer.fit(x, y, 1, nullptr);
        trainer.bind(Adam(0.001f));
        trainer.fit(x, y, 1, nullptr);
        CHECK(true, "bind worked");
    }
    OK();

    TEST("FNN generates correct architecture");
    {
        auto net = nn::FNN({4, 16, 8, 2}, nn::ReLU);
        auto& ly = net.getLayers();
        // FNN({4,16,8,2}) → Linear(4→16), ReLU, Linear(16→8), ReLU, Linear(8→2)
        CHECK(ly.size() >= 3, "has at least 3 layers");
    }
    OK();
}

// =================================[8. Gradient check with numerical grad]================================
static void test_gradcheck() {
    TEST("numerical_gradient on MSE");
    {
        Matrix<float> w(2,1); w << 0.5f << -0.3f;
        Matrix<float> t(2,1); t << 1.0f << 0.0f;
        auto loss_fn = [&t](const Matrix<float>& pred) -> float {
            return mse_loss(pred, t);
        };
        auto ng = NPCore::numerical_gradient<float>(loss_fn, w, 1e-5f);
        CHECK(ng.row == 2 && ng.col == 1, "numerical grad shape");
        // d(MSE)/dw = 2*(w-t)/N → for w=[0.5,-0.3], t=[1,0]: grad = [0.5-1, -0.3-0] = [-0.5, -0.3]
        // With /N (N=2)? No, MSE = sum((w-t)^2)/N, so grad = 2*(w-t)/N = [2*(0.5-1)/2, 2*(-0.3-0)/2] = [-0.5, -0.3]
        CLOSE(ng.at(0,0), -0.5f, 0.1f, "grad[0]");
        CLOSE(ng.at(1,0), -0.3f, 0.1f, "grad[1]");
    }
    OK();

    TEST("gradcheck passes on Linear weight gradients");
    {
        Linear lin(2, 1);
        lin.train();
        Matrix<float> x(2,1); x << 1.0f << 2.0f;
        Matrix<float> t(1,1); t << 0.5f;
        auto params = lin.getParams();
        Matrix<float> orig_w = *params[0];

        // Forward to get output
        auto out = lin.forward(x);
        int N = out.row * out.col;
        // Compute correct MSE gradient: d/dy (1/N * Σ(y_i - t_i)²) = 2*(y_i-t_i)/N
        Matrix<float> g_out_correct(out.row, out.col);
        for (int i = 0; i < N; ++i)
            g_out_correct.data_ptr()[i] = 2.0f * (out.data_ptr()[i] - t.data_ptr()[i]) / (float)N;
        auto g_input = lin.backward(g_out_correct);
        auto* analytical_w = lin.getWeightGrad();
        CHECK(analytical_w != nullptr, "weight grad allocated");
        Matrix<float> analytical_copy = *analytical_w;
        lin.CleanGard();

        // Numerical gradient
        auto loss_fn = [&lin, &x, &t](const Matrix<float>& w) -> float {
            auto pi = lin.getParams();
            *pi[0] = w;
            Matrix<float> xc(x);
            auto o = lin.forward(xc);
            float l = mse_loss(o, t);
            lin.CleanGard();
            *pi[0] = w;
            return l;
        };

        auto ng = NPCore::numerical_gradient<float>(loss_fn, orig_w, 1e-5f);
        bool ok = NPCore::gradcheck(analytical_copy, ng, 0.05f, "linear_w");
        CHECK(ok, "gradcheck passed");
    }
    OK();
}

// =================================[9. Loss functions]================================
static void test_losses() {
    TEST("cross_entropy_loss value");
    {
        Matrix<float> logits(2,3); // 2 samples, 3 classes
        logits << 2.0f << 1.0f << 0.1f;
        logits << 0.1f << 2.0f << 1.0f;
        Matrix<float> labels(2,3);
        labels << 1.0f << 0.0f << 0.0f;
        labels << 0.0f << 1.0f << 0.0f;
        float ce = cross_entropy_loss(logits, labels);
        CHECK(ce >= 0.0f, "CE loss non-negative");
    }
    OK();

    TEST("bce_loss_grad shape");
    {
        Matrix<float> p(3,1); p << 0.9f << 0.1f << 0.5f;
        Matrix<float> t(3,1); t << 1.0f << 0.0f << 1.0f;
        auto g = bce_loss_grad(p, t);
        CHECK(g.row == 3 && g.col == 1, "BCE grad shape");
    }
    OK();

    TEST("smooth_l1_loss_grad");
    {
        Matrix<float> p(2,1); p << 0.5f << 1.5f;
        Matrix<float> t(2,1); t << 0.0f << 0.0f;
        auto g = smooth_l1_loss_grad(p, t, 1.0f);
        CHECK(g.row == 2 && g.col == 1, "Huber grad shape");
        // |0.5-0|=0.5 < 1: grad = 0.5/1 = 0.5
        // |1.5-0|=1.5 >= 1: grad = sign(1.5) = 1.0
        CLOSE(g.at(0,0), 0.5f, 0.01f, "Huber grad[0] small err");
        CLOSE(g.at(1,0), 1.0f, 0.01f, "Huber grad[1] large err");
    }
    OK();
}

// =================================[10. DataLoader with SequenceLoader validation]================================
static void test_dataloader() {
    TEST("SequenceLoader dim validation");
    {
        SequenceLoader loader(3, 2, 4);
        Matrix<float> x(10, 3); // matches input_dim
        Matrix<float> y(10, 2); // matches output_dim
        loader.add_sequence(x, y);

        Matrix<float> x_bad(10, 5); // wrong input_dim
        bool threw = false;
        try { loader.add_sequence(x_bad, y); }
        catch(const std::runtime_error&) { threw = true; }
        CHECK(threw, "rejected wrong input_dim");
    }
    OK();
}

// =================================[main]================================
int main() {
    printf("=== NPCore Test Suite ===\n\n");

    printf("[1] use_bias parameter:\n");
    test_use_bias();

    printf("\n[2] Conv1d:\n");
    test_conv1d();

    printf("\n[3] TransformerEncoder:\n");
    test_transformer();

    printf("\n[4] InstanceNorm:\n");
    test_instancenorm();

    printf("\n[5] Serialization:\n");
    test_serialization();

    printf("\n[6] ONNX Export:\n");
    test_onnx();

    printf("\n[7] Trainer:\n");
    test_trainer();

    printf("\n[8] Gradient Checks:\n");
    test_gradcheck();

    printf("\n[9] Loss Functions:\n");
    test_losses();

    printf("\n[10] DataLoader:\n");
    test_dataloader();

    printf("\n=== Results: %d PASS, %d FAIL ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
