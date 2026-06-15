// =================================[Loss 函数测试]================================
#include "NPCore.h"
#include <cstdio>
#include <cmath>
using namespace NPCore;
static int passed=0,failed=0;
#define T(n) do{printf("  %-50s ... ",n);}while(0)
#define OK() do{printf("PASS\n");passed++;}while(0)
#define FAIL(m) do{printf("FAIL: %s\n",m);failed++;}while(0)
#define CHK(c,m) do{if(!(c)){FAIL(m);return;}}while(0)
#define CLOSE(a,b,eps,m) do{if(fabsf((a)-(b))>(eps)){printf("FAIL: %s (%f vs %f)\n",m,(float)(a),(float)(b));failed++;return;}}while(0)

static void t_ce(){
    T("cross_entropy_loss value");
    Matrix<float> logits(2,3);logits<<2.0f<<1.0f<<0.1f;logits<<0.1f<<2.0f<<1.0f;
    Matrix<float> labels(2,3);labels<<1.0f<<0.0f<<0.0f;labels<<0.0f<<1.0f<<0.0f;
    float ce=cross_entropy_loss(logits,labels);
    CHK(ce>=0.0f,"CE loss non-negative");OK();
}
static void t_bce_grad(){
    T("bce_loss_grad shape");
    Matrix<float> p(3,1);p<<0.9f<<0.1f<<0.5f;
    Matrix<float> t(3,1);t<<1.0f<<0.0f<<1.0f;
    auto g=bce_loss_grad(p,t);
    CHK(g.row==3&&g.col==1,"BCE grad shape");OK();
}
static void t_huber(){
    T("smooth_l1_loss_grad values");
    Matrix<float> p(2,1);p<<0.5f<<1.5f;
    Matrix<float> t(2,1);t<<0.0f<<0.0f;
    auto g=smooth_l1_loss_grad(p,t,1.0f);
    CLOSE(g.at(0,0),0.5f,0.01f,"Huber grad[0] small err");  // |0.5|<1→0.5/1
    CLOSE(g.at(1,0),1.0f,0.01f,"Huber grad[1] large err");  // |1.5|>=1→sign=1.0
    OK();
}
static void t_dataloader(){
    T("SequenceLoader dim validation");
    SequenceLoader loader(3,2,4);
    Matrix<float> x(10,3),y(10,2);loader.add_sequence(x,y);
    Matrix<float> x_bad(10,5);bool threw=false;
    try{loader.add_sequence(x_bad,y);}catch(const std::runtime_error&){threw=true;}
    CHK(threw,"rejected wrong input_dim");OK();
}
int main(){
    printf("=== Loss & Data Tests ===\n\n");
    t_ce();t_bce_grad();t_huber();t_dataloader();
    printf("\n=== Results: %d PASS, %d FAIL ===\n",passed,failed);
    return failed>0?1:0;
}
