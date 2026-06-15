// =================================[use_bias 参数测试]================================
#include "NPCore.h"
#include <cstdio>
#include <cstdlib>
using namespace NPCore;
static int passed=0,failed=0;
#define T(name) do{printf("  %-50s ... ",name);}while(0)
#define OK() do{printf("PASS\n");passed++;}while(0)
#define FAIL(m) do{printf("FAIL: %s\n",m);failed++;}while(0)
#define CHK(c,m) do{if(!(c)){FAIL(m);return;}}while(0)

static void t_linear_nobias() {
    T("Linear use_bias=false");
    Linear lin(4,8,InitMode::Uniform,false);
    CHK(lin.getParams().size()==1,"expected 1 param (weight only)");
    Matrix<float> x(4,1);x<<1<<2<<3<<4;
    auto out=lin.forward(x);
    CHK(out.row==8&&out.col==1,"output shape");
    lin.CleanGard();OK();
}
static void t_linear_bias() {
    T("Linear use_bias=true (default)");
    Linear lin(4,8);
    CHK(lin.getParams().size()==2,"expected 2 params");
    Matrix<float> x(4,1);x<<1<<2<<3<<4;
    auto out=lin.forward(x);
    CHK(out.row==8&&out.col==1,"output shape");
    lin.CleanGard();OK();
}
static void t_conv2d_nobias() {
    T("Conv2d use_bias=false");
    Conv2d conv(3,16,3,1,0,InitMode::XavierUniform,false);
    CHK(conv.getParams().size()==1,"expected 1 param");
    Matrix<float> img(32,32,3);for(int i=0;i<32*32*3;++i)img.data_ptr()[i]=0.1f;
    auto out=conv.forward(img);
    CHK(out.row==30&&out.col==30&&out.channel==16,"output shape");
    conv.CleanGard();OK();
}
static void t_conv2d_bias() {
    T("Conv2d use_bias=true (default)");
    Conv2d conv(3,16,3,1,0);
    CHK(conv.getParams().size()==2,"expected 2 params");
    Matrix<float> img(32,32,3);for(int i=0;i<32*32*3;++i)img.data_ptr()[i]=0.1f;
    auto out=conv.forward(img);
    CHK(out.row==30&&out.col==30&&out.channel==16,"output shape");
    conv.CleanGard();OK();
}
int main(){
    printf("=== use_bias Tests ===\n\n");
    t_linear_nobias();t_linear_bias();t_conv2d_nobias();t_conv2d_bias();
    printf("\n=== Results: %d PASS, %d FAIL ===\n",passed,failed);
    return failed>0?1:0;
}
