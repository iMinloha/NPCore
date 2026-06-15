// =================================[模型序列化测试]================================
#include "NPCore.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
using namespace NPCore;
static int passed=0,failed=0;
#define T(n) do{printf("  %-50s ... ",n);}while(0)
#define OK() do{printf("PASS\n");passed++;}while(0)
#define FAIL(m) do{printf("FAIL: %s\n",m);failed++;}while(0)
#define CHK(c,m) do{if(!(c)){FAIL(m);return;}}while(0)
#define CLOSE(a,b,eps,m) do{if(fabsf((a)-(b))>(eps)){printf("FAIL: %s (%.4f vs %.4f)\n",m,(float)(a),(float)(b));failed++;return;}}while(0)

static void t_save_load_weights(){
    T("save_model + load_model_weights");
    std::vector<Module<float>*> layers={new Linear(4,8),new Activation::ReLU(),new Linear(8,2)};
    Sequence m1(layers);
    CHK(save_model(m1,"test_m.np"),"save ok");
    std::vector<Module<float>*> layers2={new Linear(4,8),new Activation::ReLU(),new Linear(8,2)};
    Sequence m2(layers2);
    CHK(load_model_weights(m2,"test_m.np"),"load ok");
    auto p1=m1.getLayers()[0]->getParams(),p2=m2.getLayers()[0]->getParams();
    CHK(p1.size()==p2.size(),"same param count");
    for(int i=0;i<p1[0]->row*p1[0]->col;++i)CLOSE(p1[0]->data_ptr()[i],p2[0]->data_ptr()[i],0.001f,"weight");
    std::remove("test_m.np");OK();
}
static void t_save_load_seq(){
    T("save_sequence + load_sequence");
    std::vector<Module<float>*> layers={new Linear(3,6),new Activation::Tanh()};
    Sequence src(layers);
    std::vector<LayerTag> tags={LayerTag::Linear,LayerTag::Unknown};
    CHK(save_sequence(src,"test_s.np",tags),"save ok");
    auto loaded=load_sequence("test_s.np");
    CHK(loaded.getLayers().size()>=1,"loaded has layers");
    auto sp=src.getLayers()[0]->getParams(),lp=loaded.getLayers()[0]->getParams();
    for(int i=0;i<sp[0]->row*sp[0]->col;++i)CLOSE(sp[0]->data_ptr()[i],lp[0]->data_ptr()[i],0.001f,"weight");
    std::remove("test_s.np");OK();
}
int main(){
    printf("=== Serialization Tests ===\n\n");
    t_save_load_weights();t_save_load_seq();
    printf("\n=== Results: %d PASS, %d FAIL ===\n",passed,failed);
    return failed>0?1:0;
}
