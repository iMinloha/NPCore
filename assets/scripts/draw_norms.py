"""
Normalization Methods - Visual Comparison  (solid white bg)
Generates: assets/img/normalization.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(12, 7.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8.5)
ax.axis('off')

C = {'cyan': '#0d7377', 'blue': '#1a56db', 'purple': '#6d28d9',
     'green': '#166534', 'orange': '#b45309', 'red': '#b91c1c',
     'gray': '#6b7280', 'fg': '#111827', 'dim': '#374151', 'white': '#ffffff'}

def box(ax, x, y, w, h, color, title, sub='', fs=9.5, fs2=7):
    b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                       facecolor=color, edgecolor='none', alpha=0.22, linewidth=0, zorder=1)
    ax.add_patch(b)
    b2 = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                        facecolor='none', edgecolor=color, alpha=0.88, linewidth=1.4, zorder=2)
    ax.add_patch(b2)
    ax.text(x+w/2, y+h/2+0.04, title, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=color, fontfamily='monospace', zorder=3)
    if sub:
        ax.text(x+w/2, y+h/2-0.24, sub, ha='center', va='center', fontsize=fs2,
                color=C['dim'], fontfamily='monospace', zorder=3)

def draw_grid(ax, x, y, w, h, n_rows, n_cols, label, color, highlight, sub=''):
    cw, ch = w/n_cols, h/n_rows
    for r in range(n_rows):
        for c in range(n_cols):
            hl = (r, c) in highlight
            fc = color if hl else '#e5e7eb'
            al = 0.55 if hl else 0.18
            rect = FancyBboxPatch((x+c*cw, y+r*ch), cw*0.92, ch*0.85,
                                  boxstyle='round,pad=0.03', facecolor=fc, edgecolor='none', alpha=al, zorder=1)
            ax.add_patch(rect)
    ax.text(x+w/2, y-0.22, label, ha='center', fontsize=7, color=C['dim'],
            fontfamily='monospace', fontweight='bold', zorder=3)
    if sub:
        ax.text(x+w/2, y-0.42, sub, ha='center', fontsize=6.5, color=C['dim'], family='monospace', zorder=3)

# ── Title ──
ax.text(6, 8.10, 'Normalization - What Gets Normalized?', ha='center', fontsize=15,
        fontweight='bold', color=C['fg'], family='monospace')
ax.text(6, 7.68, 'Input tensor shape:  (Batch=N,  Height=H,  Width=W,  Channels=C)',
        ha='center', fontsize=9, color=C['dim'], family='monospace')

# ── Four methods ──
methods = [
    ('BatchNorm2d',    C['blue'],   'All (N,H,W) per channel',  'Depends on batch size',     'CNN  ·  ResNet'),
    ('LayerNorm',      C['green'],  'All channels per sample',  'Batch-size independent',     'Transformer  ·  RNN'),
    ('GroupNorm',      C['orange'], 'Channels/group + spatial', 'Batch-size independent',     'Detection  ·  GAN'),
    ('InstanceNorm',   C['purple'], 'Spatial only',             'Per sample, per channel',    'Style Transfer'),
]

y_start = 6.5
for i, (name, color, desc, batch_info, use_case) in enumerate(methods):
    y = y_start - i * 1.55
    box(ax, 0.3, y, 3.0, 0.70, color, name, '', fs=11)
    ax.text(0.5, y+0.38, desc, fontsize=7.5, color=C['dim'], fontweight='bold', family='monospace', zorder=3)
    ax.text(0.5, y+0.15, batch_info, fontsize=7.5, color=C['dim'], family='monospace', zorder=3)

    if i == 0:
        hl = {(r, c) for r in range(6) for c in range(10)}
    elif i == 1:
        hl = {(0, c) for c in range(10)}
    elif i == 2:
        hl = set()
        for g in range(4):
            for r in range(6):
                for c in range(g*2, min((g+1)*2, 8)):
                    hl.add((r, c))
    else:
        hl = {(r, c) for r in range(6) for c in range(10)}
    draw_grid(ax, 4.0, y+0.05, 2.8, 0.60, 6, 10, f'{name} - highlighted region', color, hl)

    ax.text(7.6, y+0.40, use_case, fontsize=8.5, color=color, family='monospace',
            fontweight='bold', zorder=3)

# ── Comparison table ──
t_x, t_y = 8.0, 5.6
ax.text(t_x, t_y+0.5, 'Quick Reference', fontsize=10, fontweight='bold', color=C['fg'], family='monospace', zorder=3)
rows = [
    ['Method',        'Norm Dimension',    'Batch Dependent',   'Best For'],
    ['BatchNorm2d',   '(N, H, W)',         'Yes',               'CNN / ResNet'],
    ['LayerNorm',     '(C)',               'No',                'Transformer'],
    ['GroupNorm',     '(H, W, C/G)',       'No',                'Detect / GAN'],
    ['InstanceNorm',  '(H, W)',            'No',                'Style Transfer'],
]
colors_row = [[C['fg']]*4,
              [C['blue'], C['dim'], C['dim'], C['dim']],
              [C['green'], C['dim'], C['dim'], C['dim']],
              [C['orange'], C['dim'], C['dim'], C['dim']],
              [C['purple'], C['dim'], C['dim'], C['dim']]]
for r, (row, crs) in enumerate(zip(rows, colors_row)):
    for c, (cell, cr) in enumerate(zip(row, crs)):
        fs = 7.5 if r == 0 else 7
        fw = 'bold' if r == 0 or c == 0 else 'normal'
        ax.text(t_x + c*1.1, t_y - r*0.35, cell, fontsize=fs, fontweight=fw,
                color=cr, family='monospace', zorder=3)

# ── Visual example ──
ex_y = 1.3
ax.text(0.5, ex_y+0.8, 'GroupNorm example  (C=8 channels, G=4 groups)',
        fontsize=9, fontweight='bold', color=C['fg'], family='monospace', zorder=3)
for g in range(4):
    gx = 0.5 + g * 2.9
    ax.text(gx, ex_y+0.35, f'Group {g}', fontsize=7, color=C['orange'], fontweight='bold',
            family='monospace', ha='center', zorder=3)
    for r in range(3):
        for c in range(4):
            rect = FancyBboxPatch((gx+0.2 + c*0.32, ex_y - r*0.20), 0.28, 0.16,
                                  boxstyle='round,pad=0.02',
                                  facecolor=C['orange'], edgecolor='none', alpha=0.18+r*0.08, zorder=1)
            ax.add_patch(rect)
    ax.text(gx, ex_y - 0.70, f'ch{2*g}, ch{2*g+1}', fontsize=6.5, color=C['dim'],
            fontfamily='monospace', ha='center', fontweight='bold', zorder=3)
ax.text(6, ex_y-0.45, '→  2 channels per group, normalize together over H×W×2',
        fontsize=8.5, color=C['dim'], fontfamily='monospace', fontweight='bold', zorder=3)

plt.tight_layout(pad=0.3)
plt.savefig('../img/normalization.png', dpi=180, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print('Saved: assets/img/normalization.png')
