// =================================[ONNXModel Implementation]================================
#include "Utils/ONNXModel.h"
#include "Layers/Basic/Linear.h"
#include "Layers/Basic/Flatten.h"
#include "Layers/Conv/Conv2d.h"
#include "Layers/Conv/Conv1d.h"
#include "Layers/Conv/Pool.h"
#include "Layers/Normalization/BatchNorm.h"
#include "Activations/Activation.h"
#include "Layers/Basic/Concat.h"
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <functional>
#include <unordered_map>

namespace NPCore {
namespace nn {

// =================================[Protobuf Parser]================================
struct PBReader {
    const uint8_t* p;
    const uint8_t* end;
    PBReader(const std::vector<uint8_t>& d) : p(d.data()), end(d.data()+d.size()) {}

    bool eof() const { return p >= end; }
    uint64_t varint() {
        uint64_t v=0; int s=0;
        while(p<end){ uint8_t b=*p++; v|=(uint64_t)(b&0x7F)<<s; s+=7; if(!(b&0x80))break; }
        return v;
    }
    int field_tag() { return eof()?-1:(int)varint(); }
    int field_num(int tag) { return tag>>3; }
    int wire_type(int tag) { return tag&7; }

    uint64_t read_varint() { return varint(); }
    std::string read_string() { uint64_t len=varint(); std::string s((const char*)p,(size_t)len); p+=len; return s; }
    std::vector<uint8_t> read_bytes() { uint64_t len=varint(); std::vector<uint8_t> b(p,p+len); p+=len; return b; }
    float read_f32() { float v; memcpy(&v,p,4); p+=4; return v; }
    void skip_field(int wire) {
        switch(wire){
        case 0: varint(); break;
        case 2: { uint64_t l=varint(); p+=l; break; }
        case 5: p+=4; break;
        default: break;
        }
    }
};

// =================================[Protobuf Writer]================================
struct PBWriter {
    std::string s;
    void vi(uint64_t v){ while(v>0x7F){s+=(char)((v&0x7F)|0x80);v>>=7;} s+=(char)v; }
    void fld(int n,int w){ vi((n<<3)|w); }
    void i32(int f,int32_t v){fld(f,0);vi((uint32_t)v);}
    void i64(int f,int64_t v){fld(f,0);vi((uint64_t)v);}
    void f32(int f,float v){fld(f,5);uint32_t b;memcpy(&b,&v,4);s.append((char*)&b,4);}
    void str(int f,const std::string& v){fld(f,2);vi(v.size());s+=v;}
    void bytes(int f,const void* d,size_t n){fld(f,2);vi(n);s.append((const char*)d,n);}
    void msg(int f,const std::string& m){fld(f,2);vi(m.size());s+=m;}
};

ONNXModel::ONNXModel() = default;

// =================================[LOAD from file]================================
ONNXModel ONNXModel::load(const std::string& filepath) {
    ONNXModel model;
    std::ifstream f(filepath, std::ios::binary);
    if(!f) throw std::runtime_error("ONNXModel::load: cannot open "+filepath);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)),
                               std::istreambuf_iterator<char>());
    PBReader r(data);

    // Parse ModelProto
    std::vector<uint8_t> graph_raw;
    bool has_graph=false;

    while(!r.eof()){
        int tag=r.field_tag(); if(tag<0)break;
        int fn=r.field_num(tag), wt=r.wire_type(tag);
        if(fn==1 && wt==0) model.ir_version_=(int64_t)r.read_varint();
        else if(fn==2 && wt==2) model.producer_name_=r.read_string();
        else if(fn==4 && wt==2) model.domain_=r.read_string();
        else if(fn==7 && wt==2){ graph_raw=r.read_bytes(); has_graph=true; }
        else r.skip_field(wt);
    }
    if(!has_graph) throw std::runtime_error("ONNXModel::load: no graph in file");

    // Parse GraphProto
    PBReader g(graph_raw);
    std::vector<std::vector<uint8_t>> node_raws, init_raws;

    while(!g.eof()){
        int tag=g.field_tag(); if(tag<0)break;
        int fn=g.field_num(tag), wt=g.wire_type(tag);
        if(fn==1 && wt==2) node_raws.push_back(g.read_bytes());
        else if(fn==2 && wt==2) model.graph_name_=g.read_string();
        else if(fn==5 && wt==2) init_raws.push_back(g.read_bytes());
        else if(fn==11 && wt==2){ auto b=g.read_bytes(); PBReader v(b); while(!v.eof()){int t2=v.field_tag();if(v.field_num(t2)==1)model.inputs_.push_back(v.read_string());else v.skip_field(v.wire_type(t2));} }
        else if(fn==12 && wt==2){ auto b=g.read_bytes(); PBReader v(b); while(!v.eof()){int t2=v.field_tag();if(v.field_num(t2)==1)model.outputs_.push_back(v.read_string());else v.skip_field(v.wire_type(t2));} }
        else g.skip_field(wt);
    }

    // Parse nodes
    for(auto& nr : node_raws){
        PBReader n(nr); Node node;
        while(!n.eof()){
            int tag=n.field_tag(); if(tag<0)break;
            int fn=n.field_num(tag), wt=n.wire_type(tag);
            if(fn==1 && wt==2) node.inputs.push_back(n.read_string());
            else if(fn==2 && wt==2) node.outputs.push_back(n.read_string());
            else if(fn==4 && wt==2) node.op_type=n.read_string();
            else if(fn==5 && wt==2){ // attribute
                auto ab=n.read_bytes(); PBReader a(ab);
                std::string attr_name; int attr_type=0; float fv=0; int64_t iv=0; std::vector<int64_t> isv;
                while(!a.eof()){
                    int at=a.field_tag(); if(at<0)break;
                    int af=a.field_num(at), aw=a.wire_type(at);
                    if(af==1 && aw==2) attr_name=a.read_string();
                    else if(af==2 && aw==5) fv=a.read_f32();
                    else if(af==3 && aw==0) iv=(int64_t)a.read_varint();
                    else if(af==8 && aw==0) isv.push_back((int64_t)a.read_varint());
                    else if(af==20 && aw==0) attr_type=(int)a.read_varint();
                    else a.skip_field(aw);
                }
                if(attr_type==1) node.attr_float.push_back({attr_name,fv});
                else if(attr_type==2) node.attr_int.push_back({attr_name,iv});
                else if(attr_type==7) node.attr_ints.push_back({attr_name,isv});
            }
            else n.skip_field(wt);
        }
        model.nodes_.push_back(node);
    }

    // Parse initializers
    for(auto& ir : init_raws){
        PBReader t(ir); Tensor ten;
        while(!t.eof()){
            int tag=t.field_tag(); if(tag<0)break;
            int fn=t.field_num(tag), wt=t.wire_type(tag);
            if(fn==1 && wt==0) ten.dims.push_back((int64_t)t.read_varint());
            else if(fn==2 && wt==0) t.read_varint(); // data_type, skip
            else if(fn==8 && wt==2) ten.name=t.read_string();
            else if(fn==9 && wt==2){
                auto raw=t.read_bytes();
                ten.data.resize(raw.size()/4);
                memcpy(ten.data.data(),raw.data(),raw.size());
            }
            else t.skip_field(wt);
        }
        model.weights_.push_back(ten);
    }

    return model;
}

// =================================[SAVE to file]================================
bool ONNXModel::save(const std::string& filepath) const {
    // Build NodeProtos
    std::string nodes_all, inits_all;
    int wc=0;
    for(size_t ni=0; ni<nodes_.size(); ++ni){
        auto& nd=nodes_[ni];
        PBWriter node;
        for(auto& i:nd.inputs) node.str(1,i);
        for(auto& o:nd.outputs) node.str(2,o);
        node.str(4,nd.op_type);
        for(auto& [k,v]:nd.attr_float){
            PBWriter a; a.str(1,k); a.i32(20,1); a.f32(2,v);
            PBWriter w; w.msg(5,a.s); node.s+=w.s;
        }
        for(auto& [k,v]:nd.attr_int){
            PBWriter a; a.str(1,k); a.i32(20,2); a.i64(3,v);
            PBWriter w; w.msg(5,a.s); node.s+=w.s;
        }
        for(auto& [k,v]:nd.attr_ints){
            PBWriter a; a.str(1,k); a.i32(20,7); for(auto vi:v)a.i64(8,vi);
            PBWriter w; w.msg(5,a.s); node.s+=w.s;
        }
        // Wrap as GraphProto field 1
        PBWriter nw; nw.msg(1,node.s); nodes_all+=nw.s;
    }

    // Build TensorProtos
    for(auto& wt: weights_){
        PBWriter t;
        for(auto d:wt.dims) t.i64(1,d);
        t.i32(2,1); // FLOAT
        if(!wt.name.empty()) t.str(8,wt.name);
        if(!wt.data.empty()) t.bytes(9,wt.data.data(),wt.data.size()*4);
        PBWriter iw; iw.msg(5,t.s); inits_all+=iw.s;
    }

    // ValueInfo for inputs/outputs
    auto make_vi = [](const std::string& nm)->std::string{
        PBWriter v; v.str(1,nm);
        PBWriter tt; tt.i32(1,1); // elem_type=FLOAT
        PBWriter dim; dim.str(2,"N"); // dynamic dim
        PBWriter shape; shape.msg(1,dim.s);
        tt.msg(2,shape.s);
        PBWriter type; type.msg(1,tt.s);
        v.msg(2,type.s);
        return v.s;
    };
    std::string vis;
    for(auto& in: inputs_){ PBWriter w; w.msg(11,make_vi(in)); vis+=w.s; }
    for(auto& ou: outputs_){ PBWriter w; w.msg(12,make_vi(ou)); vis+=w.s; }

    // GraphProto
    PBWriter graph;
    graph.s+=nodes_all;
    if(!graph_name_.empty()) graph.str(2,graph_name_);
    graph.s+=inits_all;
    graph.s+=vis;

    // ModelProto
    PBWriter model;
    model.i64(1, ir_version_);
    model.str(2, producer_name_);
    if(!domain_.empty()) model.str(4, domain_);
    model.msg(7, graph.s);

    // OpsetImport
    PBWriter opset;
    { PBWriter oi; oi.str(1,""); oi.i64(2,11); opset.msg(8, oi.s); }

    std::ofstream f(filepath, std::ios::binary);
    if(!f) return false;
    f.write(model.s.data(), model.s.size());
    f.write(opset.s.data(), opset.s.size());
    return f.good();
}

// =================================[EXPORT from Sequence]================================
ONNXModel ONNXModel::from_sequence(const Sequence& seq,
                                    const std::vector<int64_t>& input_shape,
                                    const std::string& graph_name) {
    ONNXModel m;
    m.graph_name_=graph_name;
    m.inputs_={"input"};
    int wc=0, nc=0;
    std::function<std::string(Module<float>*,const std::string&)> em;
    em=[&](Module<float>* L, const std::string& in)->std::string{
        std::string out="v"+std::to_string(++nc);
        Node nd; auto prm=L->getParams();
        if(auto* seq=dynamic_cast<Sequence*>(L)){ std::string p=in; for(auto* s:seq->getLayers()) p=em(s,p); return p; }
        if(auto* cat=dynamic_cast<Concat*>(L)){ std::vector<std::string> bo; for(auto* b:cat->getBranches()) bo.push_back(em(b,in)); Node cn{"Concat",bo,{out}}; cn.attr_int.push_back({"axis",1}); m.nodes_.push_back(cn); return out; }
        if(dynamic_cast<Activation::ReLU*>(L)) nd=Node{"Relu",{in},{out}};
        else if(dynamic_cast<Activation::Sigmoid*>(L)) nd=Node{"Sigmoid",{in},{out}};
        else if(dynamic_cast<Activation::Tanh*>(L)) nd=Node{"Tanh",{in},{out}};
        else if(dynamic_cast<Activation::SoftMax*>(L)){ nd=Node{"Softmax",{in},{out}}; nd.attr_int.push_back({"axis",1}); }
        else if(dynamic_cast<Activation::GELU*>(L)) nd=Node{"Gelu",{in},{out}};
        else if(dynamic_cast<MaxPool2d*>(L)){ nd=Node{"MaxPool",{in},{out}}; nd.attr_ints.push_back({"kernel_shape",{2,2}}); nd.attr_ints.push_back({"strides",{2,2}}); }
        else if(dynamic_cast<AvgPool2d*>(L)){ nd=Node{"AveragePool",{in},{out}}; nd.attr_ints.push_back({"kernel_shape",{2,2}}); nd.attr_ints.push_back({"strides",{2,2}}); }
        else if(dynamic_cast<Flatten*>(L)){ nd=Node{"Flatten",{in},{out}}; nd.attr_int.push_back({"axis",1}); }
        else if(dynamic_cast<BatchNorm1d*>(L)||dynamic_cast<BatchNorm2d*>(L)){
            std::string w="w"+std::to_string(wc++),b="w"+std::to_string(wc++),mn="w"+std::to_string(wc++),vr="w"+std::to_string(wc++);
            if(prm.size()>=2){ int N=prm[0]->row; std::vector<float> g(N),bt(N),zm(N,0),om(N,1);
                for(int i=0;i<N;++i){g[i]=prm[0]->at(i,0);bt[i]=prm[1]->at(i,0);}
                m.weights_.push_back({w,{N},g});m.weights_.push_back({b,{N},bt});m.weights_.push_back({mn,{N},zm});m.weights_.push_back({vr,{N},om}); }
            nd=Node{"BatchNormalization",{in,w,b,mn,vr},{out}};nd.attr_float.push_back({"epsilon",1e-5f});
        }else if(prm.size()>=1 && prm[0]->channel>1){
            std::string w="w"+std::to_string(wc++);int k=prm[0]->row,tc=prm[0]->channel,oc=prm.size()>=2?prm[1]->row:tc,ic=(oc>0&&tc%oc==0)?tc/oc:1;if(ic*oc!=tc){ic=1;oc=tc;}
            std::vector<float> wd(oc*ic*k*k);for(int o=0;o<oc;++o)for(int ii=0;ii<ic;++ii)for(int ki=0;ki<k;++ki)for(int kj=0;kj<k;++kj)wd[((o*ic+ii)*k+ki)*k+kj]=prm[0]->at(ki,kj,o*ic+ii);
            m.weights_.push_back({w,{oc,ic,k,k},wd});std::vector<std::string> ins={in,w};
            if(prm.size()>=2){std::string bn="w"+std::to_string(wc++);std::vector<float> bd(oc);for(int i=0;i<oc;++i)bd[i]=prm[1]->at(i,0);m.weights_.push_back({bn,{oc},bd});ins.push_back(bn);}
            nd=Node{"Conv",ins,{out}};nd.attr_ints.push_back({"kernel_shape",{k,k}});
        }else if(prm.size()>=2 && prm[1]->col==1 && prm[1]->row>=1){
            std::string w="w"+std::to_string(wc++),b="w"+std::to_string(wc++);int of=prm[0]->row,inf=prm[0]->col;
            std::vector<float> wd(inf*of),bd(of);for(int i=0;i<of;++i){for(int j=0;j<inf;++j)wd[i*inf+j]=prm[0]->at(i,j);bd[i]=prm[1]->at(i,0);}
            m.weights_.push_back({w,{of,inf},wd});m.weights_.push_back({b,{of},bd});
            nd=Node{"Gemm",{in,w,b},{out}};nd.attr_float.push_back({"alpha",1.0f});nd.attr_float.push_back({"beta",1.0f});nd.attr_int.push_back({"transB",1});
        }else if(prm.size()>=1 && prm[0]->channel==1 && prm[0]->col>1){
            std::string w="w"+std::to_string(wc++);int oc=prm[0]->row,ks=prm[0]->col;std::vector<float> wd(oc*ks);for(int o=0;o<oc;++o)for(int j=0;j<ks;++j)wd[o*ks+j]=prm[0]->at(o,j);
            m.weights_.push_back({w,{oc,1,ks},wd});std::vector<std::string> ins={in,w};
            if(prm.size()>=2){std::string bn="w"+std::to_string(wc++);std::vector<float> bd(oc);for(int i=0;i<oc;++i)bd[i]=prm[1]->at(i,0);m.weights_.push_back({bn,{oc},bd});ins.push_back(bn);}
            nd=Node{"Conv",ins,{out}};nd.attr_ints.push_back({"kernel_shape",{ks}});
        }else return in;
        if(!nd.op_type.empty())m.nodes_.push_back(nd);
        return out;
    };
    std::string prev="input";
    for(auto* L: seq.getLayers()) prev=em(L,prev);
    m.outputs_={prev};
    return m;
}

// =================================[IMPORT to Sequence]================================
Sequence ONNXModel::to_sequence() const {
    std::vector<Module<float>*> layers;
    std::unordered_map<std::string, Module<float>*> output_map;
    struct WtInfo { Tensor t; bool used=false; };
    std::vector<WtInfo> wts;
    for(auto& w:weights_) wts.push_back({w,false});

    auto find_w = [&](const std::string& nm)->Tensor*{
        for(auto& w:wts) if(w.t.name==nm){ w.used=true; return &w.t; }
        return nullptr;
    };

    for(auto& nd:nodes_){
        Module<float>* L=nullptr;

        if(nd.op_type=="Gemm"){
            auto* w=find_w(nd.inputs.size()>1?nd.inputs[1]:"");
            auto* b=find_w(nd.inputs.size()>2?nd.inputs[2]:"");
            if(!w) continue;
            for(auto& [k,v]:nd.attr_int) (void)(k=="transB" && v==1); // transB=1 uses same layout
            int of=(int)w->dims[0], inf=(int)(w->dims.size()>1?w->dims[1]:1);
            // transB=1 means ONNX computes A·B^T. B has shape [out,in].
            // NPCore Linear computes W·x where W=[out,in]. Same layout — no swap needed.
            auto* lin=new Linear(inf,of,InitMode::Uniform,b!=nullptr);
            auto pp=lin->getParams();
            for(int i=0;i<of;++i) for(int j=0;j<inf;++j) pp[0]->at(i,j)=w->data[i*inf+j];
            if(b&&pp.size()>=2) for(int i=0;i<of&&i<(int)b->data.size();++i) pp[1]->at(i,0)=b->data[i];
            L=lin;
        }
        else if(nd.op_type=="Conv"){
            auto* w=find_w(nd.inputs.size()>1?nd.inputs[1]:"");
            if(!w) continue;
            int krn=1, oc=(int)w->dims[0], ic=(int)(w->dims.size()>1?w->dims[1]:1);
            auto it_ks=std::find_if(nd.attr_ints.begin(),nd.attr_ints.end(),[](auto&p){return p.first=="kernel_shape";});
            if(it_ks!=nd.attr_ints.end() && !it_ks->second.empty()) krn=(int)it_ks->second[0];
            if(w->dims.size()>=4){ // Conv2d
                int k=(int)w->dims[2];
                auto* b=find_w(nd.inputs.size()>2?nd.inputs[2]:"");
                auto* cv=new Conv2d(ic,oc,k,1,0,InitMode::XavierUniform,b!=nullptr);
                auto pp=cv->getParams();
                for(int o=0;o<oc;++o) for(int ii=0;ii<ic;++ii)
                    for(int ki=0;ki<k;++ki) for(int kj=0;kj<k;++kj)
                        pp[0]->at(ki,kj,o*ic+ii)=w->data[((o*ic+ii)*k+ki)*k+kj];
                if(b&&pp.size()>=2) for(int i=0;i<oc&&i<(int)b->data.size();++i) pp[1]->at(i,0)=b->data[i];
                L=cv;
            } else { // Conv1d
                auto* b=find_w(nd.inputs.size()>2?nd.inputs[2]:"");
                auto* cv=new Conv1d(ic,oc,krn,1,0,InitMode::XavierUniform,b!=nullptr);
                auto pp=cv->getParams();
                for(int o=0;o<oc;++o) for(int j=0;j<(int)w->dims[2];++j) pp[0]->at(o,j)=w->data[o*(int)w->dims[2]+j];
                if(b&&pp.size()>=2) for(int i=0;i<oc&&i<(int)b->data.size();++i) pp[1]->at(i,0)=b->data[i];
                L=cv;
            }
        }
        else if(nd.op_type=="Relu") L=new Activation::ReLU();
        else if(nd.op_type=="Sigmoid") L=new Activation::Sigmoid();
        else if(nd.op_type=="Tanh") L=new Activation::Tanh();
        else if(nd.op_type=="Softmax") L=new Activation::SoftMax();
        else if(nd.op_type=="Gelu") L=new Activation::GELU();
        else if(nd.op_type=="LeakyRelu"){
            float a=0.01f; for(auto&[k,v]:nd.attr_float) if(k=="alpha")a=v;
            L=new Activation::LeakyReLU(a);
        }
        else if(nd.op_type=="MaxPool") L=new MaxPool2d(2,2);
        else if(nd.op_type=="AveragePool") L=new AvgPool2d(2,2);
        else if(nd.op_type=="Flatten"||nd.op_type=="Reshape") L=new NPCore::Flatten();
        else if(nd.op_type=="BatchNormalization"){
            auto* g=find_w(nd.inputs.size()>1?nd.inputs[1]:"");
            if(g){ int n=(int)g->dims[0]; auto* bn=new BatchNorm1d(n);
                auto pp=bn->getParams();
                for(int i=0;i<n&&i<(int)g->data.size();++i) pp[0]->at(i,0)=g->data[i];
                auto* bt=find_w(nd.inputs.size()>2?nd.inputs[2]:"");
                if(bt&&pp.size()>=2) for(int i=0;i<n&&i<(int)bt->data.size();++i) pp[1]->at(i,0)=bt->data[i];
                L=bn;
            }
        }
        else if(nd.op_type=="Concat" && nd.inputs.size()>=2){
            std::vector<Module<float>*> branches;
            for(auto& iname: nd.inputs){
                auto it=output_map.find(iname);
                if(it!=output_map.end()){
                    branches.push_back(it->second);
                    auto lit=std::find(layers.begin(),layers.end(),it->second);
                    if(lit!=layers.end())*lit=nullptr;
                }
            }
            if(!branches.empty()){
                layers.erase(std::remove(layers.begin(),layers.end(),nullptr),layers.end());
                L=new Concat(branches);
            }
        }

        if(L) layers.push_back(L);
        if(!nd.outputs.empty() && (L || !layers.empty())){
            for(auto& oname: nd.outputs) output_map[oname]=L?L:layers.back();
        }
    }
    return Sequence(layers);
}

} // namespace nn
} // namespace NPCore
