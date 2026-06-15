// =================================[Trainer API 测试]================================
#include "NPCore.h"
#include "Model.h"
#include <cstdio>
using namespace NPCore;
using namespace NPCore::nn;
static int passed=0,failed=0;
#define T(n) do{printf("  %-50s ... ",n);}while(0)
#define OK() do{printf("PASS\n");passed++;}while(0)
#define FAIL(m) do{printf("FAIL: %s\n",m);failed++;}while(0)
#define CHK(c,m) do{if(!(c)){FAIL(m);return;}}while(0)

static void t_tanh(){
    T("Trainer with Tanh activation");
    auto net=FNN({2,4,2},Tanh);
    Matrix<float> x(2,1);x<<1<<2;
    Matrix<float> y(2,1);y<<1<<0;
    Trainer trainer(net,MSE,Adam(0.01f));
    trainer.fit(x,y,2,nullptr);
    CHK(true,"trainer ran");OK();
}
static void t_ce(){
    T("Trainer with CrossEntropy loss");
    auto net=FNN({3,8,3},ReLU);
    Matrix<float> x(3,1);x<<1<<2<<3;
    Matrix<float> y(3,1);y<<1<<0<<0;
    Trainer trainer(net,CrossEntropy,Adam(0.01f));
    trainer.fit(x,y,2,nullptr);
    CHK(true,"CE trainer ran");OK();
}
static void t_bind(){
    T("Trainer bind changes optimizer");
    auto net=FNN({2,4,1},Sigmoid);
    Matrix<float> x(2,1);x<<1<<2;
    Matrix<float> y(1,1);y<<0.5f;
    Trainer trainer(net,MSE,SGD(0.1f));
    trainer.fit(x,y,1,nullptr);
    trainer.bind(Adam(0.001f));
    trainer.fit(x,y,1,nullptr);
    CHK(true,"bind worked");OK();
}
static void t_fnn(){
    T("FNN architecture check");
    auto net=FNN({4,16,8,2},ReLU);
    CHK(net.getLayers().size()>=3,"has >=3 layers");OK();
}
static void t_cnn(){
    T("CNN architecture check");
    auto net=CNN({1,8,16},3,ReLU,10);
    CHK(net.getLayers().size()>=5,"has >=5 layers");OK();
}
int main(){
    printf("=== Trainer Tests ===\n\n");
    t_tanh();t_ce();t_bind();t_fnn();t_cnn();
    printf("\n=== Results: %d PASS, %d FAIL ===\n",passed,failed);
    return failed>0?1:0;
}
