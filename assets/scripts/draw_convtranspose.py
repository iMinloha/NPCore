"""
ConvTranspose2d - Upsampling Pipeline  (solid white bg)
Generates: assets/img/convtranspose2d.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

C = {'cyan': '#0d7377', 'blue': '#1a56db', 'purple': '#6d28d9',
     'green': '#166534', 'orange': '#b45309', 'red': '#b91c1c',
     'gray': '#6b7280', 'fg': '#111827', 'dim': '#374151', 'white': '#ffffff'}

def box(ax, x, y, w, h, color, title, sub='', fs=9, fs2=7):
    b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                       facecolor=color, edgecolor='none', alpha=0.22, linewidth=0, zorder=2)
    ax.add_patch(b)
    b2 = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                        facecolor='none', edgecolor=color, alpha=0.90, linewidth=1.4, zorder=3)
    ax.add_patch(b2)
    ax.text(x+w/2, y+h/2+0.05, title, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=color, fontfamily='monospace', zorder=4)
    if sub:
        ax.text(x+w/2, y+h/2-0.22, sub, ha='center', va='center', fontsize=fs2,
                color=C['dim'], fontfamily='monospace', zorder=4)

def arrow(x1, y1, x2, y2, c=None, lw=1.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c or C['gray'], lw=lw,
                               connectionstyle='arc3,rad=0'), zorder=1)

# ── Title ──
ax.text(6, 7.55, 'ConvTranspose2d  -  Upsampling via GEMM + col2im',
        ha='center', fontsize=14, fontweight='bold', color=C['fg'], family='monospace')
ax.text(6, 7.12, 'C_in=4  →  C_out=2    kernel=3×3    stride=2    padding=1',
        ha='center', fontsize=8.5, color=C['dim'], family='monospace')
ax.text(6, 6.72, 'H_out = (H_in − 1) × stride  −  2 × padding  +  kernel  =  3×2 − 2 + 3  =  7',
        ha='center', fontsize=9, color=C['red'], family='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=C['red'], edgecolor=C['red'],
                  alpha=0.12, lw=1.0))

# ── Step 1 ──
y1 = 5.3
box(ax, 0.6, y1, 2.5, 0.58, C['blue'], 'Input', '4×4×4  tensor', fs=9)
arrow(3.2, y1+0.29, 4.3, y1+0.29, C['gray'])
box(ax, 4.4, y1, 3.2, 0.58, C['blue'], 'Flatten', '(C_in=4) × (H_in·W_in=16)', fs=8.5)

# ── Step 2 ──
y2 = 4.0
box(ax, 0.6, y2+0.08, 2.5, 0.50, C['orange'], 'Kernel', '3×3×(C_in·C_out)=3×3×8', fs=8.5)
arrow(2.2, 5.75, 1.85, y2+0.58, C['gray'], 0.8)
box(ax, 4.4, y2, 3.2, 0.66, C['purple'], 'GEMM:  Wᵀ · input_2d',
    '(C_out·K²=8) × (4,16) = (8,16)  col matrix', fs=8.5)
arrow(7.7, y1+0.29, 8.9, y1+0.29, C['gray'])
box(ax, 9.0, y2+0.08, 2.4, 0.50, C['purple'], 'Col (8×16)', '8 rows = C_out·K²')

# ── Step 3 ──
y3 = 2.5
arrow(10.2, y2, 10.2, y3+0.58, C['gray'])
box(ax, 6.5, y3, 4.5, 0.66, C['green'], 'col2im - Scatter Columns → Spatial',
    'Each input pixel → K×K patch in output  (stride=2 upsampling)', fs=8.5, fs2=7)

# ── Step 4 ──
y4 = 1.4
arrow(9.0, y3, 9.0, y4+0.58, C['gray'])
box(ax, 6.5, y4, 4.0, 0.58, C['cyan'], 'Output  +  Bias', '7×7×2  upsampled tensor')

# ── Visual hint ──
ax.text(2.2, 2.55, 'stride=2 upsampling', fontsize=7.5, color=C['dim'],
        family='monospace', ha='center', fontweight='bold')
ax.text(2.2, 2.20, '4×4  →  7×7', fontsize=9, color=C['green'], family='monospace',
        ha='center', fontweight='bold')

# ── Legend ──
leg = [(C['blue'], 'I/O'), (C['orange'], 'Weight'), (C['purple'], 'GEMM'), (C['green'], 'col2im'), (C['cyan'], 'Output')]
for i, (col, lab) in enumerate(leg):
    lx = 0.4 + i * 2.3
    ax.add_patch(FancyBboxPatch((lx, 6.95), 0.14, 0.10, boxstyle='round,pad=0.02',
                                facecolor=col, edgecolor='none', alpha=0.45, zorder=2))
    ax.text(lx+0.20, 6.97, lab, fontsize=6.5, color=C['dim'], fontweight='bold', family='monospace', zorder=3)

plt.tight_layout(pad=0.2)
plt.savefig('../img/convtranspose2d.png', dpi=180, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print('Saved: assets/img/convtranspose2d.png')
