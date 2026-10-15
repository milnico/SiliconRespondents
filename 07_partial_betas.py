"""Standardized-β analysis: discriminant validity AFTER partialling.

Addresses the §7.7 evaluative-halo limitation. Where §7.7 reports zero-order
correlations between each domain score (GRIT, SELF, LEAD, PROC) and each Big
Five trait individually, this script regresses

    domain_score ~ O + C + E + A + N

on z-scored predictors and outcome, yielding a *standardized* β for each trait.
β isolates the trait's UNIQUE linear contribution (shared variance attributed
to the trait that actually carries it), collapsing the evaluative halo that
inflates the zero-order r for "desirable" traits.

For every silicon checkpoint we already produced (silicon_outputs/<m>/seed<N>/
criterion.csv), we compute two β vectors:
  * OBSERVED (marker-based): predictors = silicon's own Big Five scores from
    the 20 IPIP marker items. This matches the human-study reporting standard.
  * LATENT (mechanistic):    predictors = the injected z values (Σ_human draws).

Outputs:
  results/betas_per_checkpoint.csv  -- long-format, one row per (ckpt × domain × predictor-type × trait).
  results/betas_summary.md          -- argmax-match contrast (r-argmax vs β-argmax); per-domain mean β±sd.

Usage:  python 07_partial_betas.py [--n_boot 2000]
"""
import argparse
import importlib.util
import os
import re
from glob import glob

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAITS = ["O", "C", "E", "A", "N"]


def _load_crit():
    spec = importlib.util.spec_from_file_location("crit", os.path.join(ROOT, "06_criterion_validity.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

crit = _load_crit()


def analyse_ckpt(silicon_csv, meta, lit, n_boot, seed_for_boot):
    """Return per-domain rows: β vectors (observed + latent) with CIs."""
    sil = pd.read_csv(silicon_csv)
    scores = crit.scale_scores(sil, meta)  # observed marker BF in [O,C,E,A,N], latent in [Oz..Nz]

    target = (meta[meta.scale_type == "domain"]
              .drop_duplicates("scale").set_index("scale")[["target_trait", "target_sign"]])
    rows = []
    for dom in lit["scale"].tolist():
        ttrait = target.loc[dom, "target_trait"]
        tsign = int(target.loc[dom, "target_sign"])

        y = scores[dom].to_numpy(dtype=float)
        for kind, cols in [("observed", TRAITS), ("latent", [t + "z" for t in TRAITS])]:
            X = scores[cols].to_numpy(dtype=float)
            beta, lo, hi, r2 = crit.betas_with_ci(X, y, n_boot, seed_for_boot)
            # also keep zero-order r for the side-by-side comparison
            zero_r = np.array([crit.corr(scores[dom], scores[c]) for c in cols])

            # argmax by |β|
            arg_i = int(np.nanargmax(np.abs(beta))) if not np.all(np.isnan(beta)) else -1
            arg_trait = TRAITS[arg_i] if arg_i >= 0 else "?"
            arg_sign = int(np.sign(beta[arg_i])) if arg_i >= 0 else 0
            ti = TRAITS.index(ttrait)
            target_beta_sign = int(np.sign(beta[ti])) if not np.isnan(beta[ti]) else 0
            row = {
                "domain": dom, "predictor_kind": kind,
                "target_trait": ttrait, "target_sign": tsign,
                "beta_argmax_trait": arg_trait, "beta_argmax_sign": arg_sign,
                "beta_argmax_match": (arg_trait == ttrait) and (arg_sign == tsign),
                "target_beta_sign_match": target_beta_sign == tsign,
                "beta_on_target": float(beta[ti]),
                "beta_on_target_ci_lo": float(lo[ti]),
                "beta_on_target_ci_hi": float(hi[ti]),
                "r2": float(r2),
            }
            for t, b, l, h, r in zip(TRAITS, beta, lo, hi, zero_r):
                row[f"beta_{t}"]    = float(b)
                row[f"beta_{t}_lo"] = float(l)
                row[f"beta_{t}_hi"] = float(h)
                row[f"r_{t}"]       = float(r)  # for the side-by-side comparison
            rows.append(row)
    return rows


def fmt_mean_sd(vals, fmt="{:+.2f}"):
    a = np.array([v for v in vals if v is not None and not np.isnan(v)])
    if len(a) == 0: return "n/a"
    if len(a) == 1: return f"{fmt.format(a[0])} (n=1)"
    return f"{fmt.format(a.mean())} ± {fmt.format(a.std(ddof=1)).lstrip('+')}"


def write_summary(df, r_detail, out_md):
    """Markdown summary: argmax-match contrast + per-domain mean β table."""
    lines = []
    lines.append("# Discriminant validity after partialling — β analysis\n")
    lines.append("Each domain score is regressed on all five Big Five predictors simultaneously "
                 "(z-scored inputs and outputs → standardized β). β-argmax isolates each trait's "
                 "*unique* contribution after the evaluative halo is partialled out.\n")

    # Argmax-match contrast table (β-argmax-match vs §7.7's r-argmax-match) per domain
    lines.append("## Argmax-match contrast (across all 7 checkpoints × 4 domains = 28 cells)\n")
    head = "| predictor | overall β-argmax-match | overall sign-match-on-target |"
    sep  = "|---|---|---|"
    lines.append(head); lines.append(sep)
    for kind in ["observed", "latent"]:
        sub = df[df.predictor_kind == kind]
        am = sub.beta_argmax_match.sum()
        sm = sub.target_beta_sign_match.sum()
        lines.append(f"| {kind} | **{am}/{len(sub)}** | {sm}/{len(sub)} |")
    # For comparison: zero-order r-argmax-match from §7.7 detail file
    if r_detail is not None and "dominant_match" in r_detail.columns:
        r_am = int(r_detail.dominant_match.sum()); r_n = len(r_detail)
        r_sm = int(r_detail.sign_match_on_target.sum())
        lines.append(f"| zero-order r (§7.7) | {r_am}/{r_n} | {r_sm}/{r_n} |")

    # Per-domain table — observed predictors, aggregated across all ckpts
    for kind in ["observed", "latent"]:
        lines.append(f"\n## Per-domain mean β ± sd (across 7 checkpoints) — predictors: {kind}\n")
        head = "| domain | target | " + " | ".join(f"β[{t}]" for t in TRAITS) + " | β-argmax match | mean R² |"
        sep  = "|" + "|".join(["---"] * (3 + len(TRAITS) + 1)) + "|"
        lines.append(head); lines.append(sep)
        for dom in ["GRIT", "SELF", "LEAD", "PROC"]:
            sub = df[(df.domain == dom) & (df.predictor_kind == kind)]
            if len(sub) == 0: continue
            ttrait = sub.target_trait.iloc[0]; tsign = int(sub.target_sign.iloc[0])
            cells = [dom, f"{tsign:+d}{ttrait}"]
            for t in TRAITS:
                cells.append(fmt_mean_sd(sub[f"beta_{t}"].tolist()))
            am = int(sub.beta_argmax_match.sum()); n = len(sub)
            cells.append(f"{am}/{n}")
            cells.append(f"{sub['r2'].mean():.2f}")
            lines.append("| " + " | ".join(cells) + " |")

    # Per-checkpoint full β tables (appendix-style) — only OBSERVED kind to keep it readable
    lines.append("\n## Per-checkpoint β (observed predictors)\n")
    head = "| model | seed | domain | target | " + " | ".join(f"β[{t}]" for t in TRAITS) + " | β-argmax | R² |"
    lines.append(head); lines.append("|" + "|".join(["---"] * (5 + len(TRAITS) + 2)) + "|")
    for _, r in df[df.predictor_kind == "observed"].iterrows():
        ttrait = r["target_trait"]; tsign = int(r["target_sign"])
        cells = [r["model"], str(int(r["seed"])), r["domain"], f"{tsign:+d}{ttrait}"]
        for t in TRAITS:
            cells.append(f"{r[f'beta_{t}']:+.2f}")
        arg = f"{r['beta_argmax_sign']:+d}{r['beta_argmax_trait']}"
        if r["beta_argmax_match"]: arg += " ✓"
        cells.append(arg)
        cells.append(f"{r['r2']:.2f}")
        lines.append("| " + " | ".join(cells) + " |")

    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")


def main(args):
    os.chdir(ROOT)
    os.makedirs("results", exist_ok=True)
    meta = pd.read_csv("data/criterion_meta.csv")
    lit = pd.read_csv("data/criterion_literature.csv")

    rows = []
    ckpts = sorted(glob("silicon_outputs/*/seed*/criterion.csv"))
    print(f"Found {len(ckpts)} criterion CSVs.")
    for csv in ckpts:
        m = re.match(r"silicon_outputs/(\w+)/seed(\d+)/criterion\.csv$", csv)
        if not m: continue
        model, seed = m.group(1), int(m.group(2))
        print(f"  analysing {model}/seed{seed} ...")
        for r in analyse_ckpt(csv, meta, lit, args.n_boot, seed_for_boot=0):
            rows.append({"model": model, "seed": seed, **r})

    df = pd.DataFrame(rows)
    df.sort_values(["model", "seed", "domain", "predictor_kind"], inplace=True)
    df.to_csv("results/betas_per_checkpoint.csv", index=False)
    print(f"Wrote results/betas_per_checkpoint.csv  ({len(df)} rows)")

    r_detail = None
    rd = "results/per_checkpoint_criterion_detail.csv"
    if os.path.exists(rd):
        r_detail = pd.read_csv(rd)
    write_summary(df, r_detail, "results/betas_summary.md")
    print("\n" + open("results/betas_summary.md").read())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_boot", type=int, default=2000)
    main(p.parse_args())
