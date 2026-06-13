"""
MultiHeadAttention - Forward Pass Diagram  (solid white bg)
Generates: assets/img/mha.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 9)
ax.axis('off')

C = {'cyan': '#0d7377', 'blue': '#1a56db', 'purple': '#6d28d9',
     'green': '#166534', 'orange': '#b45309', 'red': '#b91c1c',
     'gray': '#6b7280', 'fg': '#111827', 'dim': '#374151', 'white': '#ffffff'}

def box(ax, x, y, w, h, color, title, sub='', fs=9, fs2=6.5):
    b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                       facecolor=color, edgecolor='none', alpha=0.22, linewidth=0, zorder=1)
    ax.add_patch(b)
    b2 = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                        facecolor='none', edgecolor=color, alpha=0.92, linewidth=1.5, zorder=2)
    ax.add_patch(b2)
    ax.text(x+w/2, y+h/2+0.06, title, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=color, fontfamily='monospace', zorder=3)
    if sub:
        ax.text(x+w/2, y+h/2-0.24, sub, ha='center', va='center', fontsize=fs2,
                color=C['dim'], fontfamily='monospace', zorder=3)

def arrow(x1, y1, x2, y2, c=None, lw=1.1):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c or C['gray'], lw=lw,
                               connectionstyle='arc3,rad=0'), zorder=1)

# ── Title ──
ax.text(5, 8.55, 'MultiHeadAttention - Forward Pass', ha='center', fontsize=15,
        fontweight='bold', color=C['fg'], fontfamily='monospace')
ax.text(5, 8.10, 'd_model=12   num_heads=3   d_head=4   causal=False',
        ha='center', fontsize=8.5, color=C['dim'], family='monospace')

# ── Row 1: Input ──
y0 = 7.2
box(ax, 2.8, y0, 4.4, 0.50, C['blue'], 'Input', '(seq_len=4,  d_model=12)')

# ── Row 2: Q K V projections ──
y1 = 6.0
xs = [1.0, 3.85, 6.7]
labels = [('W_q', 'Linear (12×12)'), ('W_k', 'Linear (12×12)'), ('W_v', 'Linear (12×12)')]
for (title, sub), x in zip(labels, xs):
    box(ax, x, y1, 1.8, 0.60, C['orange'], title, sub, fs=9)
    arrow(5, y0, x+0.9, y1+0.60, C['gray'], 0.9)

# ── Row 3: Q K V tensors ──
y2 = 4.85
for name, x in [('Q', xs[0]), ('K', xs[1]), ('V', xs[2])]:
    box(ax, x, y2, 1.8, 0.50, C['orange'], name, 'reshape → (4,4,3)', fs=8.5)
    arrow(x+0.9, y1, x+0.9, y2+0.50, C['gray'], 0.8)

# ── Row 4: Attention ──
y3 = 3.45
box(ax, 1.2, y3, 7.6, 0.95, C['green'],
    'Scaled Dot-Product Attention  (× 3 heads, in parallel)',
    'scores = Kᵀ·Q / √4    →    softmax(row-wise)    →    out = V·attn',
    fs=9.5, fs2=7.5)
arrow(xs[0]+0.9, y2, 3.0, y3+0.95, C['gray'], 0.8)
arrow(xs[1]+0.9, y2, 5.0, y3+0.95, C['gray'], 0.8)
arrow(xs[2]+0.9, y2, 7.0, y3+0.95, C['gray'], 0.8)

# ── Row 5: Concat ──
y4 = 2.25
box(ax, 2.5, y4, 5.0, 0.55, C['purple'], 'Concat Heads', '(4,4,3) → (d_model=12, seq=4)')
arrow(5, y3, 5, y4+0.55, C['gray'], 0.9)

# ── Row 6: Output projection ──
y5 = 1.10
box(ax, 2.5, y5, 5.0, 0.55, C['cyan'], 'Output Projection  W_o · concat + b_o', '(12×12) · (12×4) = (12×4)')
arrow(5, y4, 5, y5+0.55, C['gray'], 0.9)

# ── Row 7: Output ──
y6 = 0.20
box(ax, 2.8, y6, 4.4, 0.50, C['blue'], 'Output', '(seq_len=4,  d_model=12)  -  same shape as Input')
arrow(5, y5, 5, y6+0.50, C['gray'], 0.9)

# ── Legend ──
leg = [(C['blue'], 'I/O'), (C['orange'], 'Projection'), (C['green'], 'Attention'),
       (C['purple'], 'Concat'), (C['cyan'], 'Output Proj')]
for i, (col, lab) in enumerate(leg):
    lx = 0.35 + i * 1.95
    ax.add_patch(FancyBboxPatch((lx, 8.77), 0.14, 0.10, boxstyle='round,pad=0.02',
                                facecolor=col, edgecolor='none', alpha=0.40, zorder=2))
    ax.text(lx+0.20, 8.80, lab, fontsize=6.5, color=C['dim'], fontweight='bold', family='monospace', zorder=3)

plt.tight_layout(pad=0.3)
plt.savefig('../img/mha.png', dpi=180, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print('Saved: assets/img/mha.png')
