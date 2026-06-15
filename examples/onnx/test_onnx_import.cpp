// =================================[ONNX Import Test — Load PyTorch-exported ONNX]================================
// Prerequisite: run  python export_torch_model.py  first
// Verifies NPCore correctly loads a complex PyTorch ONNX model with correct layers and weights.

#include "NPCore.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <vector>
using namespace NPCore;
using namespace NPCore::nn;
static int passed=0,failed=0;
#define T(n) do{printf("  %-55s ... ",n);}while(0)
#define OK() do{printf("PASS\n");passed++;}while(0)
#define FAIL(m) do{printf("FAIL: %s\n",m);failed++;}while(0)
#define CHK(c,m) do{if(!(c)){FAIL(m);return;}}while(0)

static void t_load_model(){
    T("load PyTorch ONNX model (17 nodes, 8 weights)");
    auto onnx=ONNXModel::load("torch_cnn.onnx");
    CHK(onnx.num_nodes()==17,"17 nodes");
    CHK((int)onnx.weights().size()==8,"8 weights");
    OK();
}
static void t_to_sequence(){
    T("convert ONNX to Sequence");
    auto onnx=ONNXModel::load("torch_cnn.onnx");
    auto seq=onnx.to_sequence();
    CHK(seq.getLayers().size()>=8,">=8 layers reconstructed");
    OK();
}
static void t_layer_types(){
    T("verify layer types (2 Conv, 2 ReLU, 2 MaxPool, 2 Gemm, 1 Softmax)");
    auto onnx=ONNXModel::load("torch_cnn.onnx");
    int nconv=0,nrelu=0,npool=0,ngemm=0,nsoft=0;
    for(auto& nd:onnx.nodes()){
        if(nd.op_type=="Conv") nconv++;
        else if(nd.op_type=="Relu") nrelu++;
        else if(nd.op_type=="MaxPool") npool++;
        else if(nd.op_type=="Gemm") ngemm++;
        else if(nd.op_type=="Softmax") nsoft++;
    }
    CHK(nconv==2,"2 Conv"); CHK(nrelu==3,"3 ReLU (2 conv + 1 fc)");
    CHK(npool==2,"2 MaxPool"); CHK(ngemm==2,"2 Gemm"); CHK(nsoft==1,"1 Softmax");
    OK();
}
static void t_weight_data(){
    T("weight tensors contain non-zero data");
    auto onnx=ONNXModel::load("torch_cnn.onnx");
    auto seq=onnx.to_sequence();
    auto& layers=seq.getLayers();
    bool found_conv=false,found_fc=false;
    for(auto* L:layers){
        auto p=L->getParams();
        if(p.size()>=1 && p[0]->channel>1){ // Conv2d
            float s=0;int n=p[0]->row*p[0]->col*p[0]->channel;
            for(int i=0;i<n;++i)s+=fabsf(p[0]->data_ptr()[i]);
            CHK(s>0.01f,"conv weight non-zero");found_conv=true;
        }
        if(p.size()>=2 && p[1]->col==1 && p[0]->channel==1 && p[0]->col>1){ // Linear
            float s=0;int n=p[0]->row*p[0]->col;
            for(int i=0;i<n;++i)s+=fabsf(p[0]->data_ptr()[i]);
            CHK(s>0.01f,"linear weight non-zero");found_fc=true;
        }
    }
    CHK(found_conv&&found_fc,"all weight types verified");
    OK();
}
static void t_forward_smoke(){
    T("forward pass smoke test (single layer)");
    auto onnx=ONNXModel::load("torch_cnn.onnx");
    auto seq=onnx.to_sequence();seq.eval();
    auto& layers=seq.getLayers();
    for(auto* L:layers){
        auto p=L->getParams();
        if(p.size()>=1 && p[0]->channel>1){ // first Conv2d
            Matrix<float> img(28,28,1);
            for(int i=0;i<28*28;++i)img.data_ptr()[i]=0.1f;
            auto out=L->forward(img);
            CHK(out.row>0,"output rows>0");
            L->CleanGard();
            OK();return;
        }
    }
    FAIL("no Conv2d layer");
}
int main(){
    printf("=== ONNX Import Test (PyTorch -> NPCore) ===\n\n");
    FILE* f=fopen("torch_cnn.onnx","rb");
    if(!f){printf("SKIP: run 'python export_torch_model.py' first\n");return 0;}
    fclose(f);
    t_load_model();t_to_sequence();t_layer_types();t_weight_data();t_forward_smoke();
    printf("\n=== Results: %d PASS, %d FAIL ===\n",passed,failed);
    return failed>0?1:0;
}
