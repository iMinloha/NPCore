// =================================[TransformerEncoder 测试]================================
#include "NPCore.h"
#include <cstdio>
using namespace NPCore;
static int passed=0,failed=0;
#define T(n) do{printf("  %-50s ... ",n);}while(0)
#define OK() do{printf("PASS\n");passed++;}while(0)
#define FAIL(m) do{printf("FAIL: %s\n",m);failed++;}while(0)
#define CHK(c,m) do{if(!(c)){FAIL(m);return;}}while(0)

static void t_fwd(){
    T("TransformerEncoderLayer forward");
    TransformerEncoderLayer enc(16,2,64);
    Matrix<float> x(5,16);for(int i=0;i<80;++i)x.data_ptr()[i]=0.1f;
    auto out=enc.forward(x);
    CHK(out.row==5&&out.col==16,"output shape matches input");
    enc.CleanGard();OK();
}
static void t_bwd(){
    T("TransformerEncoderLayer backward");
    TransformerEncoderLayer enc(16,2,64);enc.train();
    Matrix<float> x(5,16);for(int i=0;i<80;++i)x.data_ptr()[i]=0.1f;
    auto out=enc.forward(x);
    Matrix<float> grad(5,16);for(int i=0;i<80;++i)grad.data_ptr()[i]=0.01f;
    auto dx=enc.backward(grad);
    CHK(dx.row==5&&dx.col==16,"gradient shape");
    CHK(enc.getAllGrads().size()>0,"gradients computed");
    enc.CleanGard();OK();
}
static void t_stack(){
    T("TransformerEncoder stack (2 layers)");
    TransformerEncoder enc(32,4,128,2);
    Matrix<float> x(10,32);for(int i=0;i<320;++i)x.data_ptr()[i]=0.1f;
    auto out=enc.forward(x);
    CHK(out.row==10&&out.col==32,"stack output shape");
    enc.CleanGard();OK();
}
static void t_pe(){
    T("PositionalEncoding + TransformerEncoder");
    std::vector<Module<float>*> layers={new PositionalEncoding(16),new TransformerEncoderLayer(16,2,64)};
    Sequence model(layers);
    Matrix<float> x(5,16);for(int i=0;i<80;++i)x.data_ptr()[i]=0.1f;
    auto out=model.forward(x);
    CHK(out.row==5&&out.col==16,"PE+TF output shape");
    OK();
}
int main(){
    printf("=== Transformer Tests ===\n\n");
    t_fwd();t_bwd();t_stack();t_pe();
    printf("\n=== Results: %d PASS, %d FAIL ===\n",passed,failed);
    return failed>0?1:0;
}
