#!/usr/bin/env python3
"""Figure 2: Ablation ladder — Tucker congruence across conditioning modes.

Data from WALKTHROUGH.md §7.5 / manuscript §5.3 (matched 3 epochs, all modes).
Output: paper/figures/fig2_ablation_bars.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pathlib

pathlib.Path('paper/figures').mkdir(parents=True, exist_ok=True)

mode_labels = [
    'No traits\n(floor)',
    'Persona prompt\n(verbal)',
    'Persona prompt\n(numeric)',
    'Prefix + LoRA\n(no gate)',
    'Prefix +\ngated LoRA\n(full)',
]
ipip_vals = [0.284, 0.464, 0.496, 0.856, 0.842]
bfi_vals  = [0.284, 0.468, 0.486, 0.834, 0.858]

x = np.arange(len(mode_labels))
w = 0.35
c_ipip = '#2C6E9B'
c_bfi  = '#C07020'

fig, ax = plt.subplots(figsize=(8.2, 4.5))

b1 = ax.bar(x - w/2, ipip_vals, w,
            label='IPIP held-out items (T = 0.6)', color=c_ipip, alpha=0.88, zorder=3)
b2 = ax.bar(x + w/2, bfi_vals, w,
            label='BFI OOD inventory (T = 1.0)',   color=c_bfi,  alpha=0.88, zorder=3)

# Dotted reference lines
ax.axhline(0.284, ls=':', lw=0.9, color='#AAAAAA', zorder=2)
ax.axhline(0.496, ls=':', lw=0.9, color='#AAAAAA', zorder=2)
ax.text(4.73, 0.258, 'floor ≈ 0.28',
        ha='right', va='bottom', fontsize=7.5, color='#888888', style='italic')
ax.text(4.73, 0.508, 'persona ceiling ≈ 0.50',
        ha='right', va='bottom', fontsize=7.5, color='#888888', style='italic')

# Value labels above each bar
for b in b1:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, h + 0.006, f'{h:.3f}',
            ha='center', va='bottom', fontsize=7.0, color=c_ipip, fontweight='bold')
for b in b2:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, h + 0.006, f'{h:.3f}',
            ha='center', va='bottom', fontsize=7.0, color=c_bfi, fontweight='bold')

ax.set_ylabel('Tucker congruence', fontsize=11)
ax.set_ylim(0, 1.03)
ax.set_xticks(x)
ax.set_xticklabels(mode_labels, fontsize=8.5)
ax.legend(fontsize=9, framealpha=0.9, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, alpha=0.3, zorder=0)

plt.tight_layout()
out = 'paper/figures/fig2_ablation_bars.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved {out}')
