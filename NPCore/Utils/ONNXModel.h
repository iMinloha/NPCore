#ifndef NPCORE_UTILS_ONNXMODEL_H
#define NPCORE_UTILS_ONNXMODEL_H

#include <string>
#include <vector>
#include <cstdint>
#include "Core/Export.h"
#include "Core/Matrix.hpp"
#include "Layers/Sequence.h"

namespace NPCore {
namespace nn {

// =================================[ONNXModel — ONNX Import/Export]================================
//
// OOP design:
//   auto onnx = nn::ONNXModel::load("model.onnx");     // parse ONNX file
//   Sequence model = onnx.to_sequence();                // convert to NPCore layers
//
//   auto onnx2 = nn::ONNXModel::from_sequence(seq, {1,28,28});  // export
//   onnx2.save("exported.onnx");                                  // write to file

class NPCORE_API ONNXModel {
public:
    // --- Node descriptor ---
    struct Node {
        std::string op_type;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        std::vector<std::pair<std::string, float>>      attr_float;
        std::vector<std::pair<std::string, int64_t>>     attr_int;
        std::vector<std::pair<std::string, std::vector<int64_t>>> attr_ints;
    };

    // --- Weight tensor ---
    struct Tensor {
        std::string name;
        std::vector<int64_t> dims;
        std::vector<float> data;
    };

    // --- Constructors ---
    ONNXModel();
    ~ONNXModel() = default;

    // ========== File I/O ==========
    static ONNXModel load(const std::string& filepath);
    bool save(const std::string& filepath) const;

    // ========== Export from Sequence ==========
    static ONNXModel from_sequence(const Sequence& seq,
                                    const std::vector<int64_t>& input_shape,
                                    const std::string& graph_name = "NPCoreModel");

    // ========== Import to Sequence ==========
    Sequence to_sequence() const;

    // ========== Model info ==========
    int64_t ir_version() const { return ir_version_; }
    const std::string& producer_name() const { return producer_name_; }
    const std::string& graph_name() const { return graph_name_; }
    const std::string& domain() const { return domain_; }
    int num_nodes() const { return (int)nodes_.size(); }

    const std::vector<Node>&   nodes()  const { return nodes_; }
    const std::vector<Tensor>& weights() const { return weights_; }
    const std::vector<std::string>& inputs()  const { return inputs_; }
    const std::vector<std::string>& outputs() const { return outputs_; }

    // Mutable access for builder
    void set_ir_version(int64_t v) { ir_version_ = v; }
    void set_producer_name(const std::string& s) { producer_name_ = s; }
    void set_graph_name(const std::string& s) { graph_name_ = s; }
    void add_node(const Node& n) { nodes_.push_back(n); }
    void add_weight(const Tensor& t) { weights_.push_back(t); }
    void add_input(const std::string& s) { inputs_.push_back(s); }
    void add_output(const std::string& s) { outputs_.push_back(s); }

private:
    int64_t ir_version_ = 8;
    std::string producer_name_ = "NPCore";
    std::string domain_ = "ai.onnx";
    std::string graph_name_;
    std::vector<Node>   nodes_;
    std::vector<Tensor> weights_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
};

} // namespace nn
} // namespace NPCore

#endif
