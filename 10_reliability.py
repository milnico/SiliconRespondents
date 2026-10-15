"""Silicon test-retest reliability (§7.12).

BRM reviewers expect reliability evidence, not just validity. We already report internal
consistency (Cronbach alpha, via the R psychometrics). This script adds *test-retest
reliability*: hold the respondent population FIXED (same latents z) and re-administer the
instrument TWICE under fresh categorical-sampling noise, then correlate the two scale scores
across respondents.

Because z is fixed, this isolates the reliability contributed by the *generative sampling
step* (the only stochastic part of administration) — a clean upper-bound-style "generative
stability" reliability, distinct from human occasion-to-occasion retest (which also includes
state fluctuation). It ties directly to §7.10: temperature controls per-item marginal spread,
so it bounds how much sampling noise can attenuate retest r.

Design per (checkpoint, instrument):
  * draw ONE fixed latent population -> a .npy reused by every administration (silicon_sample.py
    --latents_file), so all administrations share the same respondents;
  * administration A: --sample_seed 101 ; administration B: --sample_seed 202
    (different sampling noise, identical respondents and items);
  * identity sanity: administration A' reuses --sample_seed 101 and must reproduce A exactly
    (model forward is deterministic in eval; only the sampling RNG varies) -> retest r == 1.0.

Reported per factor: scale-score test-retest r (Pearson, across respondents), mean item-level
retest r (key-agnostic), and Cronbach alpha per administration (sanity vs the R values).

Usage:
  PY=/home/nac/anaconda3/envs/llm_train/bin/python
  $PY 10_reliability.py            # qwen/seed0 + llama/seed1 x {IPIP, BFI}, n=500
  $PY 10_reliability.py --n 300 --no_sanity
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

PY = sys.executable
TRAITS = ["O", "C", "E", "A", "N"]
RESP_COL = "samp_resp"  # silicon_sample.py writes samp_resp (== true_resp == pred_resp)

# (name, checkpoint, model_name, items_file, keys_source, temperature, scale_max)
_QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
_LLAMA = "unsloth/llama-3-8b-instruct-bnb-4bit"
_IPIP_ITEMS = "data/item_holdout.parquet"
_BFI_ITEMS = "data/bfi_items.parquet"
_BFI_KEYS = "bfi_human_long.csv"

DEFAULT_CONFIGS = [
    # Qwen seeds 0-4
    ("qwen_seed0_ipip",  "checkpoints/qwen/seed0/simulator_full.pt", _QWEN,  _IPIP_ITEMS, _IPIP_ITEMS, 1.0, 5),
    ("qwen_seed0_bfi",   "checkpoints/qwen/seed0/simulator_full.pt", _QWEN,  _BFI_ITEMS,  _BFI_KEYS,   1.0, 6),
    ("qwen_seed1_ipip",  "checkpoints/qwen/seed1/simulator_full.pt", _QWEN,  _IPIP_ITEMS, _IPIP_ITEMS, 1.0, 5),
    ("qwen_seed1_bfi",   "checkpoints/qwen/seed1/simulator_full.pt", _QWEN,  _BFI_ITEMS,  _BFI_KEYS,   1.0, 6),
    ("qwen_seed2_ipip",  "checkpoints/qwen/seed2/simulator_full.pt", _QWEN,  _IPIP_ITEMS, _IPIP_ITEMS, 1.0, 5),
    ("qwen_seed2_bfi",   "checkpoints/qwen/seed2/simulator_full.pt", _QWEN,  _BFI_ITEMS,  _BFI_KEYS,   1.0, 6),
    ("qwen_seed3_ipip",  "checkpoints/qwen/seed3/simulator_full.pt", _QWEN,  _IPIP_ITEMS, _IPIP_ITEMS, 1.0, 5),
    ("qwen_seed3_bfi",   "checkpoints/qwen/seed3/simulator_full.pt", _QWEN,  _BFI_ITEMS,  _BFI_KEYS,   1.0, 6),
    ("qwen_seed4_ipip",  "checkpoints/qwen/seed4/simulator_full.pt", _QWEN,  _IPIP_ITEMS, _IPIP_ITEMS, 1.0, 5),
    ("qwen_seed4_bfi",   "checkpoints/qwen/seed4/simulator_full.pt", _QWEN,  _BFI_ITEMS,  _BFI_KEYS,   1.0, 6),
    # Llama seeds 1-4
    ("llama_seed1_ipip", "checkpoints/llama/seed1/simulator_full.pt", _LLAMA, _IPIP_ITEMS, _IPIP_ITEMS, 1.0, 5),
    ("llama_seed1_bfi",  "checkpoints/llama/seed1/simulator_full.pt", _LLAMA, _BFI_ITEMS,  _BFI_KEYS,   1.0, 6),
    ("llama_seed2_ipip", "checkpoints/llama/seed2/simulator_full.pt", _LLAMA, _IPIP_ITEMS, _IPIP_ITEMS, 1.0, 5),
    ("llama_seed2_bfi",  "checkpoints/llama/seed2/simulator_full.pt", _LLAMA, _BFI_ITEMS,  _BFI_KEYS,   1.0, 6),
    ("llama_seed3_ipip", "checkpoints/llama/seed3/simulator_full.pt", _LLAMA, _IPIP_ITEMS, _IPIP_ITEMS, 1.0, 5),
    ("llama_seed3_bfi",  "checkpoints/llama/seed3/simulator_full.pt", _LLAMA, _BFI_ITEMS,  _BFI_KEYS,   1.0, 6),
    ("llama_seed4_ipip", "checkpoints/llama/seed4/simulator_full.pt", _LLAMA, _IPIP_ITEMS, _IPIP_ITEMS, 1.0, 5),
    ("llama_seed4_bfi",  "checkpoints/llama/seed4/simulator_full.pt", _LLAMA, _BFI_ITEMS,  _BFI_KEYS,   1.0, 6),
]

OUT_DIR = "silicon_outputs/reliability"


def load_keys(keys_source):
    """item_id -> (factor, reverse). Works for the IPIP parquet and the BFI long CSV."""
    df = pd.read_parquet(keys_source) if keys_source.endswith(".parquet") else pd.read_csv(keys_source)
    df = df[["item_id", "factor", "reverse"]].drop_duplicates("item_id")
    return df.set_index("item_id").to_dict("index")


def administer(cfg_name, checkpoint, model_name, items_file, temperature, n, sample_seed, latents_file,
               out_suffix=""):
    """Run silicon_sample.py for one administration; return the long-format DataFrame.

    Idempotent: skips the GPU call if the output CSV already exists. The identity-check pass passes
    out_suffix="_identity" so it doesn't collide with administration A's file on disk.
    """
    out = os.path.join(OUT_DIR, f"{cfg_name}_s{sample_seed}{out_suffix}.csv")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"  [skip] {out} exists ({os.path.getsize(out)} bytes)")
        return pd.read_csv(out)
    cmd = [PY, "silicon_sample.py",
           "--checkpoint", checkpoint, "--model_name", model_name,
           "--items_file", items_file, "--output_file", out,
           "--n", str(n), "--temperature", str(temperature),
           "--seed", "0",                       # fixed latent-draw seed -> same respondents
           "--sample_seed", str(sample_seed),   # varies the response sampling only
           "--latents_file", latents_file]      # created on first call, loaded thereafter
    print(f"  [administer] {cfg_name} sample_seed={sample_seed}{out_suffix} -> {out}")
    subprocess.run(cmd, check=True)
    return pd.read_csv(out)


def scale_scores(long_df, keys, scale_max):
    """Wide respondent x factor scale scores (reverse-coded item means)."""
    df = long_df.copy()
    df["factor"] = df["item_id"].map(lambda i: keys[i]["factor"])
    df["rev"] = df["item_id"].map(lambda i: bool(keys[i]["reverse"]))
    r = df[RESP_COL].astype(float)
    df["scored"] = np.where(df["rev"], (scale_max + 1) - r, r)
    return df.pivot_table(index="participant_id", columns="factor", values="scored", aggfunc="mean")


def item_wide(long_df):
    """Wide respondent x item raw responses (key-agnostic; used for item-level retest)."""
    return long_df.pivot_table(index="participant_id", columns="item_id",
                               values=RESP_COL, aggfunc="mean")


def cronbach_alpha(long_df, keys, scale_max, factor):
    """Standardised-item Cronbach alpha for one factor on a single administration."""
    df = long_df.copy()
    df["factor"] = df["item_id"].map(lambda i: keys[i]["factor"])
    df["rev"] = df["item_id"].map(lambda i: bool(keys[i]["reverse"]))
    df = df[df["factor"] == factor]
    r = df["samp_resp"].astype(float)
    df["scored"] = np.where(df["rev"], (scale_max + 1) - r, r)
    mat = df.pivot_table(index="participant_id", columns="item_id", values="scored").values
    k = mat.shape[1]
    if k < 2:
        return float("nan")
    item_var = mat.var(axis=0, ddof=1).sum()
    total_var = mat.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return (k / (k - 1)) * (1 - item_var / total_var)


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def run_config(cfg, n, do_sanity):
    name, ckpt, model_name, items_file, keys_source, temp, scale_max = cfg
    if not os.path.exists(ckpt):
        print(f"  SKIP {name}: checkpoint missing ({ckpt})")
        return []
    keys = load_keys(keys_source)
    latents_file = os.path.join(OUT_DIR, f"{name}_latents.npy")

    a = administer(name, ckpt, model_name, items_file, temp, n, 101, latents_file)
    b = administer(name, ckpt, model_name, items_file, temp, n, 202, latents_file)

    sa, sb = scale_scores(a, keys, scale_max), scale_scores(b, keys, scale_max)
    common = sa.index.intersection(sb.index)
    sa, sb = sa.loc[common], sb.loc[common]
    ia, ib = item_wide(a).loc[common], item_wide(b).loc[common]

    identity_r = float("nan")
    if do_sanity:
        # Re-run sample_seed=101 to a DIFFERENT file so we genuinely re-execute the silicon
        # sampler (skip-if-exists won't conflate it with A) and can later verify identity_r ≈ 1.0.
        c = administer(name, ckpt, model_name, items_file, temp, n, 101, latents_file,
                       out_suffix="_identity")
        sc = scale_scores(c, keys, scale_max).loc[common]
        identity_r = np.nanmean([pearson(sa[f], sc[f]) for f in sa.columns if f in sc.columns])

    rows = []
    for f in TRAITS:
        if f not in sa.columns or f not in sb.columns:
            continue
        items_f = [i for i in ia.columns if keys.get(i, {}).get("factor") == f]
        item_rs = [pearson(ia[i], ib[i]) for i in items_f]
        rows.append({
            "config": name, "factor": f,
            "retest_r_scale": round(pearson(sa[f], sb[f]), 3),
            "mean_item_retest_r": round(float(np.nanmean(item_rs)), 3) if item_rs else float("nan"),
            "n_items": len(items_f),
            "alpha_A": round(cronbach_alpha(a, keys, scale_max, f), 3),
            "alpha_B": round(cronbach_alpha(b, keys, scale_max, f), 3),
            "identity_r": round(identity_r, 4) if np.isfinite(identity_r) else "n/a",
            "n_resp": len(common),
        })
    return rows


def write_summary(df, path):
    lines = ["# Silicon test-retest reliability (§7.12)\n"]
    lines.append("Respondent population held FIXED (same latents z); instrument re-administered "
                 "twice under fresh response-sampling noise (sample_seed 101 vs 202). "
                 "`retest_r_scale` = Pearson r of reverse-coded scale scores across respondents; "
                 "`mean_item_retest_r` = mean per-item retest r (key-agnostic); "
                 "`identity_r` = scale-score r when the second administration reuses sample_seed 101 "
                 "(determinism check, should be 1.0).\n")
    for cfg in df["config"].unique():
        sub = df[df["config"] == cfg]
        lines.append(f"\n## {cfg}  (n={sub['n_resp'].iloc[0]})\n")
        lines.append("| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(f"| {r.factor} | {r.retest_r_scale} | {r.mean_item_retest_r} | "
                         f"{r.n_items} | {r.alpha_A} | {r.alpha_B} | {r.identity_r} |")
        lines.append(f"\nMean scale-score retest r: **{sub['retest_r_scale'].mean():.3f}**.")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def write_aggregate(df, path):
    """Write group-level mean ± SD across seeds for each (backbone, inventory, factor)."""
    def _parse(config):
        # e.g. "qwen_seed3_ipip" -> ("qwen", "3", "ipip")
        parts = config.split("_")
        backbone = parts[0]
        seed = parts[1].replace("seed", "")
        inv = parts[2]
        return backbone, seed, inv

    rows = []
    df = df.copy()
    df["backbone"], df["seed"], df["inventory"] = zip(*df["config"].map(_parse))

    for (backbone, inventory), grp in df.groupby(["backbone", "inventory"]):
        # trimmed: exclude qwen seed3 (bad-init outlier)
        trimmed = grp[~((grp.backbone == "qwen") & (grp.seed == "3"))]
        for factor in TRAITS:
            fgrp = grp[grp.factor == factor]
            ftrim = trimmed[trimmed.factor == factor]
            if fgrp.empty:
                continue
            rows.append({
                "backbone": backbone, "inventory": inventory, "factor": factor,
                "n_seeds_all": len(fgrp),
                "retest_r_mean_all": round(fgrp.retest_r_scale.mean(), 3),
                "retest_r_std_all":  round(fgrp.retest_r_scale.std(ddof=1), 3),
                "alpha_mean_all":    round(((fgrp.alpha_A + fgrp.alpha_B) / 2).mean(), 3),
                "alpha_std_all":     round(((fgrp.alpha_A + fgrp.alpha_B) / 2).std(ddof=1), 3),
                "n_seeds_trim": len(ftrim),
                "retest_r_mean_trim": round(ftrim.retest_r_scale.mean(), 3) if len(ftrim) else float("nan"),
                "retest_r_std_trim":  round(ftrim.retest_r_scale.std(ddof=1), 3) if len(ftrim) > 1 else float("nan"),
                "alpha_mean_trim":    round(((ftrim.alpha_A + ftrim.alpha_B) / 2).mean(), 3) if len(ftrim) else float("nan"),
                "alpha_std_trim":     round(((ftrim.alpha_A + ftrim.alpha_B) / 2).std(ddof=1), 3) if len(ftrim) > 1 else float("nan"),
            })
    agg = pd.DataFrame(rows)
    agg.to_csv(path, index=False)
    print(f"Wrote {path}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--no_sanity", action="store_true", help="skip the identity (determinism) check")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="subset of config names to run (default: all)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    configs = DEFAULT_CONFIGS
    if args.configs:
        configs = [c for c in DEFAULT_CONFIGS if c[0] in args.configs]

    all_rows = []
    for cfg in configs:
        print(f"\n=== {cfg[0]} ===")
        all_rows += run_config(cfg, args.n, do_sanity=not args.no_sanity)

    df = pd.DataFrame(all_rows)
    df.to_csv("results/reliability.csv", index=False)
    print(f"\nWrote results/reliability.csv ({len(df)} rows)")
    write_summary(df, "results/reliability.md")
    write_aggregate(df, "results/reliability_aggregate.csv")


if __name__ == "__main__":
    main()
