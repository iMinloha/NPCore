// =================================[ONNX Model 导入/导出测试]================================
// 导出后可运行:  python verify_onnx.py
#include "NPCore.h"
#include <cstdio>
#include <fstream>
using namespace NPCore;
using namespace NPCore::nn;
static int passed=0,failed=0;
#define T(n) do{printf("  %-50s ... ",n);}while(0)
#define OK() do{printf("PASS\n");passed++;}while(0)
#define FAIL(m) do{printf("FAIL: %s\n",m);failed++;}while(0)
#define CHK(c,m) do{if(!(c)){FAIL(m);return;}}while(0)

static bool file_ok(const char* path){
    std::ifstream f(path,std::ios::binary);
    if(!f)return false;
    f.seekg(0,std::ios::end);
    return f.tellg()>50;
}

// ============ 导出测试 ============
static void t_export_mlp(){
    T("export MLP (Linear->ReLU->Linear->Softmax)");
    std::vector<Module<float>*> layers={new Linear(2,8),new Activation::ReLU(),new Linear(8,2),new Activation::SoftMax()};
    Sequence seq(layers);
    auto onnx=ONNXModel::from_sequence(seq,{1,2},"MLP");
    CHK(onnx.save("test_mlp.onnx"),"save ok");
    CHK(file_ok("test_mlp.onnx"),"file non-empty");
    OK();
}
static void t_export_cnn(){
    T("export CNN (Conv2d->ReLU->MaxPool->Flatten->Linear)");
    std::vector<Module<float>*> layers={new Conv2d(1,8,3,1,0),new Activation::ReLU(),new MaxPool2d(2,2),new NPCore::Flatten(),new Linear(13*13*8,10)};
    Sequence seq(layers);
    auto onnx=ONNXModel::from_sequence(seq,{1,1,28,28},"CNN");
    CHK(onnx.save("test_cnn.onnx"),"save ok");
    CHK(file_ok("test_cnn.onnx"),"file non-empty");
    OK();
}
static void t_export_bn(){
    T("export BatchNorm (Linear->BN->ReLU)");
    std::vector<Module<float>*> layers={new Linear(4,8),new BatchNorm1d(8),new Activation::ReLU()};
    Sequence seq(layers);
    auto onnx=ONNXModel::from_sequence(seq,{1,4},"BN");
    CHK(onnx.save("test_bn.onnx"),"save ok");
    CHK(file_ok("test_bn.onnx"),"file non-empty");
    OK();
}

// ============ 导入测试 ============
static void t_roundtrip(){
    T("roundtrip: export → load → to_sequence");
    std::vector<Module<float>*> layers={new Linear(3,6),new Activation::ReLU(),new Linear(6,2)};
    Sequence src(layers);
    auto onnx=ONNXModel::from_sequence(src,{1,3},"RT");
    CHK(onnx.save("test_rt.onnx"),"save ok");

    auto loaded=ONNXModel::load("test_rt.onnx");
    CHK(loaded.num_nodes()==3,"3 nodes");
    CHK(loaded.weights().size()==4,"4 weights (2W+2B)");
    CHK(loaded.graph_name()=="RT","graph name preserved");

    auto seq2=loaded.to_sequence();
    CHK(seq2.getLayers().size()>=2,"reconstructed layers");
    std::remove("test_rt.onnx");OK();
}
static void t_onnx_info(){
    T("load ONNX and inspect metadata");
    auto onnx=ONNXModel::load("test_mlp.onnx");
    CHK(onnx.ir_version()==8,"ir_version");
    CHK(onnx.producer_name()=="NPCore","producer");
    CHK(onnx.num_nodes()==4,"4 nodes");
    CHK(!onnx.inputs().empty(),"has inputs");
    CHK(!onnx.outputs().empty(),"has outputs");
    for(auto& n:onnx.nodes()) printf("\n    %s: in=%zu out=%zu",n.op_type.c_str(),n.inputs.size(),n.outputs.size());
    printf("\n");OK();
}
int main(){
    printf("=== ONNX Model Tests ===\n\n");
    printf("Outputs: test_mlp.onnx, test_cnn.onnx, test_bn.onnx\n");
    printf("Verify:  python verify_onnx.py\n\n");
    t_export_mlp();t_export_cnn();t_export_bn();
    t_roundtrip();t_onnx_info();
    printf("\n=== Results: %d PASS, %d FAIL ===\n",passed,failed);
    return failed>0?1:0;
}
