#ifndef NPCORE_UTILS_SERIALIZER_H
#define NPCORE_UTILS_SERIALIZER_H

#include <string>
#include <vector>
#include "Core/Matrix.hpp"
#include "Layers/Module.h"
#include "Layers/Sequence.h"

namespace NPCore {

// =================================[Serializer — Model Save/Load]================================
// Binary format:
//   Magic:  "NPCM" (4 bytes)
//   Version: uint32_t
//   Num layers: uint32_t
//   For each layer:
//     Layer type tag: uint32_t
//     Num params: uint32_t
//     For each param:
//       rows, cols, channels: int32_t each
//       data: float[rows*cols*channels]

// Layer type tags (must match between save and load)
enum class LayerTag : uint32_t {
    Linear = 1,
    Conv2d = 2,
    ConvTranspose2d = 3,
    RNN = 4,
    LSTM = 5,
    GRU = 6,
    BatchNorm1d = 7,
    BatchNorm2d = 8,
    LayerNorm = 9,
    GroupNorm = 10,
    MultiHeadAttention = 11,
    Embedding = 12,
    Conv1d = 13,
    InstanceNorm1d = 14,
    InstanceNorm2d = 15,
    Unknown = 999
};

NPCORE_API bool save_matrix(const Matrix<float>& mat, std::ofstream& out);
NPCORE_API Matrix<float> load_matrix(std::ifstream& in);

NPCORE_API bool save_sequence(const Sequence& seq, const std::string& filepath,
                               const std::vector<LayerTag>& tags);
NPCORE_API Sequence load_sequence(const std::string& filepath);

// Convenience: auto-save all params of any Sequence (uses Unknown tag for all)
NPCORE_API bool save_model(const Sequence& seq, const std::string& filepath);
NPCORE_API bool load_model_weights(Sequence& seq, const std::string& filepath);

} // namespace NPCore

#endif
