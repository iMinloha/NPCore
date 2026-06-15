// =================================[Conv1d 测试]================================
#include "NPCore.h"
#include <cstdio>
using namespace NPCore;
static int passed=0,failed=0;
#define T(n) do{printf("  %-50s ... ",n);}while(0)
#define OK() do{printf("PASS\n");passed++;}while(0)
#define FAIL(m) do{printf("FAIL: %s\n",m);failed++;}while(0)
#define CHK(c,m) do{if(!(c)){FAIL(m);return;}}while(0)

static void t_fwd_shape(){
    T("Conv1d forward shape");
    Conv1d conv(3,8,3,1,0);
    Matrix<float> seq(10,3);for(int i=0;i<30;++i)seq.data_ptr()[i]=0.1f;
    auto out=conv.forward(seq);
    CHK(out.row==8&&out.col==8,"output (L-K+1=8, out_ch=8)");
    conv.CleanGard();OK();
}
static void t_fwd_bwd(){
    T("Conv1d forward+backward");
    Conv1d conv(2,4,3,1,0);conv.train();
    Matrix<float> x(6,2);for(int i=0;i<12;++i)x.data_ptr()[i]=(i+1)*0.1f;
    auto out=conv.forward(x);
    Matrix<float> grad(out.row,out.col);for(int i=0;i<grad.row*grad.col;++i)grad.data_ptr()[i]=0.01f;
    auto dx=conv.backward(grad);
    CHK(dx.row==6&&dx.col==2,"grad shape matches input");
    CHK(conv.getWeightGrad()!=nullptr,"weight grad computed");
    conv.CleanGard();OK();
}
static void t_nobias(){
    T("Conv1d use_bias=false");
    Conv1d conv(2,4,3,1,0,InitMode::XavierUniform,false);
    CHK(conv.getParams().size()==1,"weight only");
    conv.CleanGard();OK();
}
int main(){
    printf("=== Conv1d Tests ===\n\n");
    t_fwd_shape();t_fwd_bwd();t_nobias();
    printf("\n=== Results: %d PASS, %d FAIL ===\n",passed,failed);
    return failed>0?1:0;
}
