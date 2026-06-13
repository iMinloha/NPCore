"""
CorePP — Library Architecture Overview  (transparent bg)
Generates: assets/img/architecture.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.set_xlim(0, 15)
ax.set_ylim(0, 11.5)
ax.axis('off')

C = {'cyan': '#0d7377', 'blue': '#1a56db', 'purple': '#6d28d9',
     'green': '#166534', 'orange': '#b45309', 'red': '#b91c1c',
     'gray': '#9ca3af', 'fg': '#111827', 'dim': '#4b5563', 'white': '#ffffff'}

def box(ax, x, y, w, h, color, title, sub=None, fs=9.5, fs2=7):
    b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                       facecolor=color, edgecolor='none', alpha=0.08, linewidth=0, zorder=1)
    ax.add_patch(b)
    b2 = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                        facecolor='none', edgecolor=color, alpha=0.55, linewidth=1.0, zorder=2)
    ax.add_patch(b2)
    ax.text(x+w/2, y+h/2, title, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=color, fontfamily='monospace', zorder=3)
    if sub:
        for i, s in enumerate(sub):
            ax.text(x+w/2, y+h-0.28-i*0.26, s, ha='center', va='top', fontsize=fs2,
                    color=C['dim'], fontfamily='monospace', zorder=3)

def section(ax, x, y, w, h, color, label, items, fs=9):
    b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.10',
                       facecolor=color, edgecolor='none', alpha=0.06, linewidth=0, zorder=1)
    ax.add_patch(b)
    b2 = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.10',
                        facecolor='none', edgecolor=color, alpha=0.50, linewidth=0.9, zorder=2)
    ax.add_patch(b2)
    ax.text(x+w/2, y+h-0.18, label, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=color, fontfamily='monospace', zorder=3)
    for i, item in enumerate(items):
        ax.text(x+w/2, y+h-0.42-i*0.24, item, ha='center', va='top', fontsize=6.2,
                color=C['dim'], fontfamily='monospace', zorder=3)

# ── Header ──
ax.text(7.5, 11.05, 'CorePP  Architecture', ha='center', fontsize=20,
        fontweight='bold', color=C['fg'], family='monospace')
ax.text(7.5, 10.55, 'C++20 Deep Learning Library    ·    AVX2 SIMD & CUDA GPU Backends',
        ha='center', fontsize=9, color=C['dim'], family='monospace')

# ═══════════ Row 1: API Layer ═══════════
y1 = 9.2
section(ax, 0.3, y1, 3.8, 1.05, C['cyan'], 'Model.h — High-Level API',
         ['nn::FNN({4,8,16,8,4}, Sigmoid)  —  Fully Connected',
          'nn::CNN({3,4,2}, 3, Sigmoid, 4) —  Convolutional',
          'nn::Trainer(net, loss, optim).fit(x,y,epochs,cb)'])
section(ax, 4.5, y1, 3.2, 1.05, C['cyan'], 'Autograd.h',
         ['numerical_gradient()  —  central difference',
          'gradcheck()  —  analytical vs numerical'])
section(ax, 8.1, y1, 3.4, 1.05, C['cyan'], 'DataLoader.h',
         ['DataLoader base class  —  Train / Test / Val',
          'InMemoryLoader  —  vector-based datasets',
          'next_batch(x, y)  —  epoch-aware iteration'])
section(ax, 11.8, y1, 2.9, 1.05, C['gray'], 'Utils',
         ['Timer  —  high-res clock',
          'RandomGenerator  —  thread-safe'])

# ═══════════ Row 2: Layers ═══════════
y2 = 7.4
section(ax, 0.3, y2, 1.7, 1.55, C['green'], 'Basic',
         ['Linear', 'Flatten', 'Embedding', 'Dropout'])
section(ax, 2.3, y2, 2.8, 1.55, C['green'], 'Convolution',
         ['Conv2d  —  im2col + GEMM',
          'ConvTranspose2d  —  upsampling',
          'MaxPool2d  /  AvgPool2d',
          'AdaptiveAvgPool2d  —  global pool'])
section(ax, 5.4, y2, 1.8, 1.55, C['green'], 'Recurrent',
         ['RNN  —  BPTT', 'LSTM  —  4-gate fused', 'GRU  —  2-gate'])
section(ax, 7.5, y2, 2.4, 1.55, C['green'], 'Attention',
         ['MultiHeadAttention',
          'Q / K / V  projections',
          'Scaled Dot-Product',
          'Causal mask support'])
section(ax, 10.2, y2, 2.4, 1.55, C['green'], 'Normalization',
         ['BatchNorm1d  /  BatchNorm2d',
          'LayerNorm', 'GroupNorm'])
section(ax, 12.9, y2, 1.8, 1.55, C['green'], 'Architecture',
         ['Residual', 'ResNetBlock', 'Sequence'])

# ═══════════ Row 3: Optimizers & Loss ═══════════
y3 = 5.7
section(ax, 0.3, y3, 5.2, 1.40, C['orange'], 'Optimizers (10)',
         ['SGD    Momentum    Adam    AdamW    RMSProp    Adagrad    Adadelta    NAdam    RAdam',
          'CosineLR  /  StepLR  —  learning rate schedules',
          'GradientClipping  —  clip_by_norm  /  clip_by_value'])
section(ax, 5.8, y3, 4.0, 1.40, C['orange'], 'Loss Functions (6)',
         ['MSE      MAE (L1)      SmoothL1 (Huber)',
          'CrossEntropy      BinaryCrossEntropy',
          'KL Divergence'])
section(ax, 10.1, y3, 4.6, 1.40, C['purple'], 'ParamInit  &  Core Utilities',
         ['Weight Init:  Zeros  Ones  Uniform  Gaussian',
          'XavierUniform    HeNormal    Orthogonal',
          'ConvUtils:  im2col / col2im  ·  Winograd F(2,3)',
          'LinearAlgebra:  Gram-Schmidt orthogonalization'])

# ═══════════ Row 4: Core Engine ═══════════
y4 = 3.5
section(ax, 0.3, y4, 3.0, 1.90, C['blue'], 'Matrix<T>  —  Tensor Engine',
         ['2D (row,col)  &  3D (row,col,channel)',
          'Operator overloading:  +  −  *  /  <<',
          'GPU transfer:  .cuda()  /  .cpu()',
          'Fused optimizer updates  (SIMD)',
          'Analysis()  —  smart formatting',
          'Gram-Schmidt  orthogonalization'])
section(ax, 3.6, y4, 4.8, 1.90, C['blue'], 'GEMM Engine  —  Matrix Multiply',
         ['GotoBLAS-style packed buffer micro-kernel',
          '6×16 AVX2 tile  —  256-bit SIMD instructions',
          'Kahan compensated summation  —  numerical stability',
          'Block-tiled L1/L2 cache optimization',
          'OpenMP multi-threaded parallelism',
          '8.0 GFLOPS  @  1024×1024  (i7-12700H)'])
section(ax, 8.7, y4, 2.8, 1.90, C['red'], 'CUDA Backend',
         ['32×32 tiled GEMM kernel',
          'Element-wise: sigmoid / tanh / relu',
          'RNN & LSTM cell kernels',
          'Auto-detect  ·  MinGW-compatible',
          'model.cuda()  —  one-liner GPU'])
section(ax, 11.8, y4, 2.9, 1.90, C['purple'], 'Activations (12)',
         ['ReLU      LeakyReLU      Sigmoid',
          'Tanh      SoftMax        ELU',
          'SELU      Softplus       Mish',
          'GELU      Swish',
          'Each:  forward + backward',
          'OpenMP parallel  (large mats)'])

# ═══════════ Row 5: Hardware ═══════════
y5 = 2.2
section(ax, 0.3, y5, 14.4, 0.95, C['gray'], 'Hardware Acceleration & Build System',
         ['AVX2 (256-bit SIMD)    ·    ARM NEON    ·    OpenMP Multi-threading    ·    CUDA GPU (optional)',
          'CMake 3.18+    ·    MinGW GCC 12+ / MSVC 2022    ·    C++20    ·    Static (.a) & Shared (.dll) builds'])

# ═══════════ Row 6: Stats ═══════════
y6 = 1.0
stats = [('25+', 'Layers'), ('12', 'Activations'), ('6', 'Losses'),
         ('10', 'Optimizers'), ('14', 'Tests'), ('7', 'Docs')]
for i, (num, label) in enumerate(stats):
    sx = 1.5 + i * 2.5
    ax.text(sx, y6+0.45, num, ha='center', fontsize=18, fontweight='bold',
            color=C['fg'], family='monospace')
    ax.text(sx, y6+0.08, label, ha='center', fontsize=8, color=C['dim'], family='monospace')

# connecting lines
for yt, yb in [(y1, y2+1.55), (y2, y3+1.40), (y3, y4+1.90), (y4, y5+0.95)]:
    ax.plot([7.5, 7.5], [yt, yb], color=C['gray'], lw=0.6, alpha=0.25, zorder=0)

plt.tight_layout(pad=0.3)
plt.savefig('../img/architecture.png', dpi=180, bbox_inches='tight', transparent=True, edgecolor='none')
plt.close()
print('Saved: assets/img/architecture.png')
