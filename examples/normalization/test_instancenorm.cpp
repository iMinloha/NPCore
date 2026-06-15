// =================================[InstanceNorm 测试]================================
#include "NPCore.h"
#include <cstdio>
using namespace NPCore;
static int passed=0,failed=0;
#define T(n) do{printf("  %-50s ... ",n);}while(0)
#define OK() do{printf("PASS\n");passed++;}while(0)
#define FAIL(m) do{printf("FAIL: %s\n",m);failed++;}while(0)
#define CHK(c,m) do{if(!(c)){FAIL(m);return;}}while(0)

static void t_in1d_fwd(){
    T("InstanceNorm1d forward");
    InstanceNorm1d inorm(8);
    Matrix<float> x(4,8);for(int i=0;i<32;++i)x.data_ptr()[i]=(float)(i+1);
    auto out=inorm.forward(x);
    CHK(out.row==4&&out.col==8,"output shape");
    inorm.CleanGard();OK();
}
static void t_in1d_bwd(){
    T("InstanceNorm1d backward");
    InstanceNorm1d inorm(8);inorm.train();
    Matrix<float> x(4,8);for(int i=0;i<32;++i)x.data_ptr()[i]=(float)(i+1);
    auto out=inorm.forward(x);
    Matrix<float> grad(4,8);for(int i=0;i<32;++i)grad.data_ptr()[i]=0.1f;
    auto dx=inorm.backward(grad);
    CHK(dx.row==4&&dx.col==8,"grad shape");
    inorm.CleanGard();OK();
}
static void t_in2d_fwd(){
    T("InstanceNorm2d forward");
    InstanceNorm2d in2d(3);
    Matrix<float> img(8,8,3);for(int i=0;i<192;++i)img.data_ptr()[i]=(i+1)*0.01f;
    auto out=in2d.forward(img);
    CHK(out.row==8&&out.col==8&&out.channel==3,"output shape");
    in2d.CleanGard();OK();
}
int main(){
    printf("=== InstanceNorm Tests ===\n\n");
    t_in1d_fwd();t_in1d_bwd();t_in2d_fwd();
    printf("\n=== Results: %d PASS, %d FAIL ===\n",passed,failed);
    return failed>0?1:0;
}
