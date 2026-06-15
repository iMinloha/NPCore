// =================================[ONNX Import/Export Tests]================================
#include "NPCore.h"
#include <cstdio>
#include <fstream>
using namespace NPCore;
using namespace NPCore::nn;
static int ok=0,bad=0;
#define T(n) do{printf("  %-50s ",n);}while(0)
#define OK() do{printf("PASS\n");ok++;}while(0)
#define FAIL(m) do{printf("FAIL: %s\n",m);bad++;}while(0)
#define CHK(c,m) do{if(!(c)){FAIL(m);return;}}while(0)

static bool fok(const char* path){std::ifstream ff(path,std::ios::binary);ff.seekg(0,std::ios::end);return ff.good()&&ff.tellg()>50;}

// ============ Export ============
static void t_export_mlp(){
    T("export MLP");
    Sequence seq({new Linear(2,8),new Activation::ReLU(),new Linear(8,2),new Activation::SoftMax()});
    auto onnx=ONNXModel::from_sequence(seq,{1,2},"MLP");
    CHK(onnx.save("test_mlp.onnx"),"save");CHK(fok("test_mlp.onnx"),"file");OK();
}
static void t_export_cnn(){
    T("export CNN");
    Sequence seq({new Conv2d(1,8,3,1,0),new Activation::ReLU(),new MaxPool2d(2,2),new NPCore::Flatten(),new Linear(13*13*8,10)});
    auto onnx=ONNXModel::from_sequence(seq,{1,1,28,28},"CNN");
    CHK(onnx.save("test_cnn.onnx"),"save");CHK(fok("test_cnn.onnx"),"file");OK();
}
static void t_export_nested(){
    T("export nested Sequence");
    auto* inner=new Sequence({new Linear(2,4),new Activation::ReLU()});
    Sequence outer({inner,new Linear(4,1)});
    auto onnx=ONNXModel::from_sequence(outer,{1,2},"Nested");
    CHK(onnx.save("test_nest.onnx"),"save");CHK(onnx.num_nodes()==3,"3 nodes flattened");OK();
}
static void t_export_concat(){
    T("export Concat");
    Concat cat({new Conv2d(1,3,3,1,1),new Conv2d(1,4,5,1,2)});
    Sequence seq({&cat});
    auto onnx=ONNXModel::from_sequence(seq,{1,1,8,8},"Concat");
    CHK(onnx.save("test_concat.onnx"),"save");OK();
}

// ============ Import ============
static void t_load(){
    T("load ONNX + inspect");
    auto onnx=ONNXModel::load("test_mlp.onnx");
    CHK(onnx.num_nodes()==4,"4 nodes");CHK(onnx.weights().size()==4,"4 weights");
    CHK(onnx.producer_name()=="NPCore","producer");CHK(onnx.ir_version()==8,"ir_version");
    OK();
}
static void t_roundtrip(){
    T("roundtrip: export → load → to_sequence");
    Sequence src({new Linear(3,6),new Activation::ReLU(),new Linear(6,2)});
    auto onnx=ONNXModel::from_sequence(src,{1,3},"RT");onnx.save("test_rt.onnx");
    auto loaded=ONNXModel::load("test_rt.onnx");auto seq=loaded.to_sequence();
    CHK(seq.getLayers().size()>=2,"layers reconstructed");
    auto p=seq.getParams();CHK(!p.empty()&&p[0]->row>0,"weights loaded");OK();
}
static void t_import_conv(){
    T("import ONNX Conv → Conv2d");
    auto onnx=ONNXModel::load("test_cnn.onnx");
    auto seq=onnx.to_sequence();seq.eval();
    Matrix<float> img(28,28,1);for(int i=0;i<28*28;i++)img.data_ptr()[i]=0.1f;
    auto out=seq.getLayers()[0]->forward(img);
    CHK(out.row>0,"conv forward ok");OK();
}

int main(){
    printf("=== ONNX Tests ===\n\n");
    printf("Output: test_mlp.onnx, test_cnn.onnx, test_nest.onnx, test_concat.onnx\n");
    printf("Verify: python verify_onnx.py\n\n");
    t_export_mlp();t_export_cnn();t_export_nested();t_export_concat();
    t_load();t_roundtrip();t_import_conv();
    // Cleanup
    std::remove("test_mlp.onnx");std::remove("test_cnn.onnx");
    std::remove("test_nest.onnx");std::remove("test_concat.onnx");std::remove("test_rt.onnx");
    printf("\n=== %d PASS, %d FAIL ===\n",ok,bad);
    return bad>0?1:0;
}
