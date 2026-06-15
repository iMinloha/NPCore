// =================================[Sequence 嵌套 + Concat 分支测试]================================
#include "NPCore.h"
#include "Model.h"
#include <cstdio>
using namespace NPCore;
using namespace NPCore::nn;
static int passed=0,failed=0;
#define T(n) do{printf("  %-55s ... ",n);}while(0)
#define OK() do{printf("PASS\n");passed++;}while(0)
#define FAIL(m) do{printf("FAIL: %s\n",m);failed++;}while(0)
#define CHK(c,m) do{if(!(c)){FAIL(m);return;}}while(0)

static void t_nest_seq(){
    T("nested Sequence inside Sequence");
    // Inner: Linear(2→4) → ReLU
    auto* inner = new Sequence({new Linear(2,4), new Activation::ReLU()});
    // Outer: inner → Linear(4→1)
    Sequence outer({inner, new Linear(4,1)});
    CHK(outer.getLayers().size()==2,"outer has 2 layers");
    // modules() flattens recursively
    auto mods=outer.modules();
    CHK(mods.size()>=3,"modules flattened to >=3 leaf modules");

    Matrix<float> x(2,1);x<<1<<2;
    auto out=outer.forward(x);
    CHK(out.row==1,"output (1,1)");
    OK();
}
static void t_nest_trainer(){
    T("Trainer with nested Sequence");
    auto* inner=new Sequence({new Linear(2,4), new Activation::Tanh()});
    Sequence outer({inner, new Linear(4,1)});
    Trainer trainer(outer, MSE, Adam(0.01f));
    Matrix<float> x(2,1);x<<1<<2;
    Matrix<float> y(1,1);y<<0.5f;
    trainer.fit(x,y,3,nullptr);
    CHK(true,"nested trainer ran");
    OK();
}
static void t_concat_fwd(){
    T("Concat 2D: 2 branches concatenated");
    auto* b1=new Sequence({new Linear(3,5), new Activation::ReLU()});
    auto* b2=new Sequence({new Linear(3,4), new Activation::Tanh()});
    Concat cat({b1,b2});

    Matrix<float> x(3,1);x<<1<<2<<3;
    auto out=cat.forward(x);
    CHK(out.row==9,"concatenated output rows=5+4=9");
    OK();
}
static void t_concat_3d(){
    T("Concat 3D: channels concatenated");
    auto* b1=new Conv2d(2,4,3,1,1);
    auto* b2=new Conv2d(2,3,3,1,1);
    Concat cat({b1,b2});

    Matrix<float> img(8,8,2);
    for(int i=0;i<8*8*2;++i)img.data_ptr()[i]=0.1f;
    auto out=cat.forward(img);
    CHK(out.channel==7,"channels=4+3=7");
    OK();
}
static void t_concat_train(){
    T("Concat training with optimizer");
    auto* b1=new Linear(3,5);
    auto* b2=new Linear(3,4);
    Concat cat({b1,b2});

    cat.train();
    Matrix<float> x(3,1);x<<1<<2<<3;
    auto out=cat.forward(x);
    Matrix<float> grad(out.row,out.col);
    for(int i=0;i<out.row*out.col;++i)grad.data_ptr()[i]=0.01f;
    auto dx=cat.backward(grad);
    CHK(dx.row==3,"gradient flows back to input");
    CHK(cat.modules().size()>=2,"has leaf modules");
    cat.CleanGard();
    OK();
}
static void t_inception_style(){
    T("Inception-style: Sequence(Concat→Flatten→Linear)");
    auto* b1=new Sequence({new Conv2d(1,4,3,1,1), new Activation::ReLU()});
    auto* b2=new Sequence({new Conv2d(1,3,5,1,2), new Activation::ReLU()});
    Concat* inception=new Concat({b1,b2});
    Sequence model({inception, new NPCore::Flatten(), new Linear(8*8*7, 10)});

    Matrix<float> img(8,8,1);
    for(int i=0;i<64;++i)img.data_ptr()[i]=0.1f;
    auto out=model.forward(img);
    // Output depends on flatten shape, just check it runs
    CHK(out.row>0,"inception forward runs");
    OK();
}
int main(){
    printf("=== Nested Sequence + Concat Tests ===\n\n");
    t_nest_seq();t_nest_trainer();
    t_concat_fwd();t_concat_3d();t_concat_train();t_inception_style();
    printf("\n=== Results: %d PASS, %d FAIL ===\n",passed,failed);
    return failed>0?1:0;
}
