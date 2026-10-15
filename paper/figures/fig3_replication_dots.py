#!/usr/bin/env python3
"""Figure 3: Tucker congruence across seeds and backbones (replication).

Reads:  results/per_checkpoint.csv
Output: paper/figures/fig3_replication_dots.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import pathlib

pathlib.Path('paper/figures').mkdir(parents=True, exist_ok=True)

df = pd.read_csv('results/per_checkpoint.csv')

c_qwen  = '#2C6E9B'
c_llama = '#C07020'

# (model, seed, panel) → annotate in that panel
# Qwen seed 3: both panels; Llama seed 4: BFI only
OUTLIER_PANELS = {
    ('qwen',  3, 'ipip'): True,
    ('qwen',  3, 'bfi'):  True,
    ('llama', 4, 'bfi'):  True,
}

panels = [
    ('ipip_congruence', 'ipip', 'IPIP held-out items'),
    ('bfi_congruence',  'bfi',  'BFI OOD inventory'),
]

fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.5))

for ax, (col, panel_key, title) in zip(axes, panels):
    for model, x_center, color, marker in [
        ('qwen',  0, c_qwen,  'o'),
        ('llama', 1, c_llama, '^'),
    ]:
        sub = df[df.model == model].sort_values('seed').reset_index(drop=True)
        n = len(sub)
        offsets = np.linspace(-0.18, 0.18, n)

        for i, row in sub.iterrows():
            key = (model, int(row.seed), panel_key)
            is_outlier = key in OUTLIER_PANELS
            ax.scatter(x_center + offsets[i - sub.index[0]], row[col],
                       c=color, marker=marker, s=60, zorder=5,
                       alpha=0.6 if is_outlier else 0.92,
                       edgecolors='#333333' if is_outlier else 'none',
                       linewidths=0.8)
            if is_outlier:
                xp = x_center + offsets[i - sub.index[0]]
                ax.annotate(
                    f's{int(row.seed)}*',
                    xy=(xp, row[col]),
                    xytext=(xp + 0.11, row[col] - 0.025),
                    fontsize=7.5, color='#555555',
                    arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=0.7),
                )

        # Trimmed mean/SD: exclude Qwen seed 3 (bad-init outlier)
        healthy = sub[sub.seed != 3][col] if model == 'qwen' else sub[col]
        m, s = healthy.mean(), healthy.std()

        # Mean bar + SD whiskers
        ax.hlines(m, x_center - 0.22, x_center + 0.22, color=color, lw=2.2, zorder=4)
        ax.vlines(x_center, m - s, m + s, color=color, lw=1.5, zorder=4)
        ax.hlines([m - s, m + s], x_center - 0.06, x_center + 0.06,
                  color=color, lw=1.2, zorder=4)

        # Mean value annotation
        ax.text(x_center + 0.24, m, f'{m:.3f}',
                va='center', fontsize=7.5, color=color, fontweight='bold')

    ax.set_title(title, fontsize=10)
    ax.set_ylabel('Tucker congruence', fontsize=10)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Qwen-2.5-1.5B\n(n = 5 seeds)', 'Llama-3-8B\n(n = 4 seeds)'], fontsize=9.5)
    ax.set_ylim(0.60, 1.00)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)

legend_handles = [
    mlines.Line2D([], [], color=c_qwen,  marker='o', markersize=7, lw=0, label='Qwen seed'),
    mlines.Line2D([], [], color=c_llama, marker='^', markersize=7, lw=0, label='Llama seed'),
    mlines.Line2D([], [], color='gray', lw=2.0, label='Trimmed mean ± SD'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=9,
           framealpha=0.9, bbox_to_anchor=(0.5, -0.05))

axes[0].text(0.5, -0.18,
             '* Annotated outliers: Qwen s3 = bad-init basin (val loss 2.28); '
             'Llama s4 = BFI OOD dip.',
             transform=axes[0].transAxes, fontsize=7, color='#666666', ha='left')

plt.suptitle('Tucker congruence across seeds and backbones', fontsize=11, y=1.02)
plt.tight_layout()
out = 'paper/figures/fig3_replication_dots.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved {out}')
