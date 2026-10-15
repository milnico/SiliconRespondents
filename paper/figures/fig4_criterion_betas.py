#!/usr/bin/env python3
"""Figure 4: Discriminant validity — standardized beta profiles per criterion domain,
with literature meta-analytic zero-order r values overlaid on a secondary axis.

The primary (left) y-axis shows silicon standardized β from multivariate regression.
The secondary (right) y-axis shows the meta-analytic zero-order r from the literature.
Both axes are scaled symmetrically around 0, so the shared zero line is visually aligned.

Reads:  results/betas_per_checkpoint.csv
        data/criterion_literature.csv
Output: paper/figures/fig4_criterion_betas.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import pathlib

pathlib.Path('paper/figures').mkdir(parents=True, exist_ok=True)

df  = pd.read_csv('results/betas_per_checkpoint.csv')
lit = pd.read_csv('data/criterion_literature.csv', index_col='scale')

# Observed predictors only; exclude bad-init Qwen seed 3
df = df[
    (df.predictor_kind == 'observed') &
    ~((df.model == 'qwen') & (df.seed == 3))
]
n_ckpts = df[['model', 'seed']].drop_duplicates().shape[0]

traits    = ['O', 'C', 'E', 'A', 'N']
beta_cols = [f'beta_{t}' for t in traits]

# Domain order matches manuscript §5.4 table
domains = ['GRIT', 'LEAD', 'SELF', 'PROC']
domain_meta = {
    'GRIT': {'title': 'Grit  (target: +C)',           'target': 'C'},
    'LEAD': {'title': 'Leadership  (target: +E)',      'target': 'E'},
    'SELF': {'title': 'Self-esteem  (target: −N)',     'target': 'N'},
    'PROC': {'title': 'Procrastination  (target: −C)', 'target': 'C'},
}

c_target = '#D4A020'   # amber/gold for the target trait bar
c_other  = '#CCCCCC'   # neutral gray for non-target bars
c_lit    = '#CC3333'   # dark red for literature reference line

# Symmetric y-axis limits: zero lines align across both axes
YLIM_BETA = (-0.70, 0.70)   # primary axis (silicon β)
YLIM_LIT  = (-1.0,  1.0)    # secondary axis (literature r)

fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))  # sharey handled manually
axes_flat = axes.flatten()

for idx, (ax, domain) in enumerate(zip(axes_flat, domains)):
    sub   = df[df.domain == domain]
    means = sub[beta_cols].mean().values
    sds   = sub[beta_cols].std().values

    target = domain_meta[domain]['target']
    colors = [c_target if t == target else c_other for t in traits]

    x = np.arange(len(traits))

    # --- Primary axis: silicon standardized β bars ---
    ax.bar(x, means, yerr=sds, capsize=4,
           color=colors, alpha=0.92, zorder=3,
           error_kw=dict(elinewidth=1.0, ecolor='#666666', capthick=1.0))
    ax.axhline(0, color='#333333', lw=0.9, zorder=2)
    ax.set_ylim(*YLIM_BETA)
    ax.set_xticks(x)
    ax.set_xticklabels(traits, fontsize=10)
    ax.set_ylabel('Silicon standardized β', fontsize=9)
    ax.set_title(domain_meta[domain]['title'], fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.25, zorder=0)

    # Annotate the target bar's mean value
    tidx  = traits.index(target)
    m_t   = means[tidx]
    sd_t  = sds[tidx]
    yann  = m_t + sd_t + 0.04 if m_t >= 0 else m_t - sd_t - 0.04
    vaann = 'bottom' if m_t >= 0 else 'top'
    ax.text(tidx, yann, f'{m_t:+.2f}',
            ha='center', va=vaann, fontsize=8.5, color=c_target, fontweight='bold')

    # R² annotation (top-right for negative-target panels, bottom-right otherwise)
    r2    = sub['r2'].mean()
    r2_y  = 0.97 if m_t < 0 else 0.04
    r2_va = 'top'  if m_t < 0 else 'bottom'
    ax.text(0.97, r2_y, r'$R^2$' + f' = {r2:.2f}', transform=ax.transAxes,
            fontsize=8, ha='right', va=r2_va, color='#555555')

    # --- Secondary axis: literature zero-order r ---
    lit_r = lit.loc[domain, traits].values.astype(float)

    ax2 = ax.twinx()
    ax2.plot(x, lit_r, color=c_lit, lw=1.8, ls='--', zorder=6,
             marker='D', markersize=5,
             markerfacecolor='white', markeredgecolor=c_lit, markeredgewidth=1.5)
    ax2.set_ylim(*YLIM_LIT)
    ax2.spines['top'].set_visible(False)

    # Show right-axis labels and ticks only on right-column panels
    right_col = (idx % 2 == 1)
    if right_col:
        ax2.set_ylabel('Literature zero-order r', fontsize=8.5, color=c_lit)
        ax2.spines['right'].set_color(c_lit)
        ax2.tick_params(axis='y', colors=c_lit, labelsize=7.5)
    else:
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(axis='y', length=0, labelsize=0)

    # Annotate the literature target-trait value
    lit_target = lit_r[tidx]
    yann_lit   = lit_target + 0.06 if lit_target >= 0 else lit_target - 0.06
    vaann_lit  = 'bottom' if lit_target >= 0 else 'top'
    ax2.text(tidx, yann_lit, f'r={lit_target:+.2f}',
             ha='center', va=vaann_lit, fontsize=7.5, color=c_lit, style='italic')

# Shared legend
legend_handles = [
    mpatches.Patch(color=c_target, label='Target trait — silicon β (left axis)'),
    mpatches.Patch(color=c_other,  label='Non-target traits — silicon β (left axis)'),
    mlines.Line2D([], [], color=c_lit, lw=1.8, ls='--', marker='D', markersize=5,
                  markerfacecolor='white', markeredgecolor=c_lit,
                  label='Literature zero-order r (right axis)'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=3,
           fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

plt.suptitle(
    f'Discriminant validity: silicon standardized β vs. literature zero-order r\n'
    f'(β: mean ± SD across {n_ckpts} checkpoints, observed predictors; '
    f'r: meta-analytic targets)',
    fontsize=10, y=1.01,
)
plt.tight_layout()
out = 'paper/figures/fig4_criterion_betas.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved {out}')
