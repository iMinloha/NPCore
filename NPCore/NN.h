#ifndef NPCORE_NN_H
#define NPCORE_NN_H

#include "Layers/Module.h"
#include "Layers/ParamInit.h"
#include "Layers/Sequence.h"

// Basic
#include "Layers/Basic/Linear.h"
#include "Layers/Basic/Flatten.h"
#include "Layers/Basic/Embedding.h"
#include "Layers/Basic/Dropout.h"

// Convolution
#include "Layers/Conv/Conv2d.h"
#include "Layers/Conv/ConvTranspose2d.h"
#include "Layers/Conv/Pool.h"

// Recurrent
#include "Layers/Recurrent/RNN.h"
#include "Layers/Recurrent/LSTM.h"
#include "Layers/Recurrent/GRU.h"

// Normalization
#include "Layers/Normalization/BatchNorm.h"
#include "Layers/Normalization/LayerNorm.h"
#include "Layers/Normalization/GroupNorm.h"

// Attention
#include "Layers/Attention/MultiHeadAttention.h"

// Architecture
#include "Layers/Architecture/Residual.h"
#include "Layers/Architecture/ResNetBlock.h"

// Activations
#include "Activations/Activation.h"

#endif
