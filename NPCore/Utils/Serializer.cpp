#include "Utils/Serializer.h"
#include "Layers/Basic/Linear.h"
#include "Layers/Conv/Conv2d.h"
#include "Layers/Conv/Conv1d.h"
#include "Layers/Normalization/LayerNorm.h"
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace NPCore {

static constexpr uint32_t MAGIC = 0x4D43504E; // "NPCM" in little-endian
static constexpr uint32_t VERSION = 1;

// =================================[Matrix save/load]================================

bool save_matrix(const Matrix<float>& mat, std::ofstream& out) {
    if (!out) return false;
    int r = mat.row, c = mat.col, ch = mat.channel;
    out.write(reinterpret_cast<const char*>(&r), sizeof(int));
    out.write(reinterpret_cast<const char*>(&c), sizeof(int));
    out.write(reinterpret_cast<const char*>(&ch), sizeof(int));
    int total = r * c * ch;
    out.write(reinterpret_cast<const char*>(mat.data_ptr()), total * sizeof(float));
    return out.good();
}

Matrix<float> load_matrix(std::ifstream& in) {
    int r, c, ch;
    in.read(reinterpret_cast<char*>(&r), sizeof(int));
    in.read(reinterpret_cast<char*>(&c), sizeof(int));
    in.read(reinterpret_cast<char*>(&ch), sizeof(int));
    if (!in) throw std::runtime_error("Serializer: failed to read matrix shape");

    Matrix<float> mat(r, c, ch);
    int total = r * c * ch;
    in.read(reinterpret_cast<char*>(mat.data_ptr()), total * sizeof(float));
    if (!in) throw std::runtime_error("Serializer: failed to read matrix data");
    return mat;
}

// =================================[Sequence save/load with tags]================================

bool save_sequence(const Sequence& seq, const std::string& filepath,
                   const std::vector<LayerTag>& tags) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) return false;

    out.write(reinterpret_cast<const char*>(&MAGIC), sizeof(uint32_t));
    uint32_t ver = VERSION;
    out.write(reinterpret_cast<const char*>(&ver), sizeof(uint32_t));

    const auto& layers = seq.getLayers();
    uint32_t num_layers = static_cast<uint32_t>(layers.size());
    out.write(reinterpret_cast<const char*>(&num_layers), sizeof(uint32_t));

    for (size_t li = 0; li < layers.size(); ++li) {
        uint32_t tag = static_cast<uint32_t>(li < tags.size() ? tags[li] : LayerTag::Unknown);
        out.write(reinterpret_cast<const char*>(&tag), sizeof(uint32_t));

        auto params = layers[li]->getParams();
        uint32_t np = static_cast<uint32_t>(params.size());
        out.write(reinterpret_cast<const char*>(&np), sizeof(uint32_t));

        for (auto* p : params)
            if (!save_matrix(*p, out)) return false;
    }

    return out.good();
}

Sequence load_sequence(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) throw std::runtime_error("Serializer: cannot open " + filepath);

    uint32_t magic = 0, ver = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    if (magic != MAGIC) throw std::runtime_error("Serializer: invalid file format");

    in.read(reinterpret_cast<char*>(&ver), sizeof(uint32_t));
    if (ver != VERSION) throw std::runtime_error("Serializer: unsupported version");

    uint32_t num_layers = 0;
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(uint32_t));

    std::vector<Module<float>*> layers;
    for (uint32_t li = 0; li < num_layers; ++li) {
        uint32_t tag_raw = 0;
        in.read(reinterpret_cast<char*>(&tag_raw), sizeof(uint32_t));
        LayerTag tag = static_cast<LayerTag>(tag_raw);

        uint32_t np = 0;
        in.read(reinterpret_cast<char*>(&np), sizeof(uint32_t));

        std::vector<Matrix<float>> param_mats;
        for (uint32_t pi = 0; pi < np; ++pi)
            param_mats.push_back(load_matrix(in));

        // Reconstruct layer based on tag
        Module<float>* layer = nullptr;
        switch (tag) {
        case LayerTag::Linear: {
            int out_f = param_mats[0].row, in_f = param_mats[0].col;
            bool has_bias = (np >= 2);
            auto* L = new Linear(in_f, out_f, InitMode::Uniform, has_bias);
            *L->getParams()[0] = param_mats[0];
            if (has_bias && np >= 2) *L->getParams()[1] = param_mats[1];
            layer = L;
            break;
        }
        case LayerTag::Conv2d: {
            int k = param_mats[0].row;
            int total_ch = param_mats[0].channel;
            int out_ch = param_mats.size() >= 2 ? param_mats[1].row : total_ch;
            int in_ch = total_ch / out_ch;
            if (in_ch * out_ch != total_ch) {
                // Fallback: assume 1 input channel
                in_ch = 1;
                out_ch = total_ch;
            }
            bool has_bias = (np >= 2);
            auto* L = new Conv2d(in_ch, out_ch, k, 1, 0, InitMode::XavierUniform, has_bias);
            *L->getParams()[0] = param_mats[0];
            if (has_bias && np >= 2) *L->getParams()[1] = param_mats[1];
            layer = L;
            break;
        }
        case LayerTag::Conv1d: {
            int out_ch = param_mats[0].row;
            int kC = param_mats[0].col;
            int in_ch = kC; // fallback, need kernel_size info
            bool has_bias = (np >= 2);
            // Approximate: kernel_size can't be recovered without extra metadata
            // Use default kernel_size=3
            auto* L = new Conv1d(in_ch, out_ch, 3, 1, 0, InitMode::XavierUniform, has_bias);
            *L->getParams()[0] = param_mats[0];
            if (has_bias && np >= 2) *L->getParams()[1] = param_mats[1];
            layer = L;
            break;
        }
        case LayerTag::LayerNorm: {
            int features = param_mats[0].row;
            auto* L = new LayerNorm(features);
            *L->getParams()[0] = param_mats[0];
            if (np >= 2) *L->getParams()[1] = param_mats[1];
            layer = L;
            break;
        }
        default: {
            // Unknown layer — skip (just consume the matrices)
            layer = nullptr;
            break;
        }
        }

        if (layer) layers.push_back(layer);
    }

    return Sequence(layers);
}

// =================================[Simple save/load — just weights, no type info]================================

bool save_model(const Sequence& seq, const std::string& filepath) {
    // Use default tag for all layers
    const auto& lyrs = seq.getLayers();
    std::vector<LayerTag> tags(lyrs.size(), LayerTag::Unknown);
    return save_sequence(seq, filepath, tags);
}

bool load_model_weights(Sequence& seq, const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) return false;

    uint32_t magic = 0, ver = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    if (magic != MAGIC) return false;

    in.read(reinterpret_cast<char*>(&ver), sizeof(uint32_t));
    if (ver != VERSION) return false;

    uint32_t num_layers = 0;
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(uint32_t));

    const auto& layers = seq.getLayers();
    if (num_layers != static_cast<uint32_t>(layers.size())) return false;

    for (uint32_t li = 0; li < num_layers; ++li) {
        uint32_t tag_raw = 0;
        in.read(reinterpret_cast<char*>(&tag_raw), sizeof(uint32_t)); // skip tag

        uint32_t np = 0;
        in.read(reinterpret_cast<char*>(&np), sizeof(uint32_t));

        auto params = layers[li]->getParams();
        if (np != static_cast<uint32_t>(params.size())) return false;

        for (uint32_t pi = 0; pi < np; ++pi) {
            Matrix<float> mat = load_matrix(in);
            if (mat.row == params[pi]->row && mat.col == params[pi]->col &&
                mat.channel == params[pi]->channel) {
                int n = mat.row * mat.col * mat.channel;
                for (int i = 0; i < n; ++i)
                    params[pi]->data_ptr()[i] = mat.data_ptr()[i];
            } else {
                return false;
            }
        }
    }

    return true;
}

} // namespace NPCore
