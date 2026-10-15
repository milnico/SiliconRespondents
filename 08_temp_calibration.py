"""Held-out temperature calibration (§7.10).

For each (checkpoint × eval-set × T) cell, generate a silicon sample at that
temperature and compute:
  * a CALIBRATION objective (per-item moment match vs the cal-half of humans);
    used to select T* without ever looking at the headline metric.
  * the headline psychometrics (Tucker mean congruence, Phi Frobenius error,
    item-total profile corr) against the REP-half of humans; used purely for
    the sensitivity curve and the at-T* report.

This breaks the circularity of the original "we tuned T on the same eval that
produces the headline numbers" objection. Humans are split 50/50 by a
deterministic sort+seed=42 on participant_id; silicon is regenerated per T.

Usage: python 08_temp_calibration.py
Outputs:
  results/temp_calibration.csv   long format, one row per (ckpt, eval, T).
  results/temp_calibration.md    summary with T*, sensitivity curve, comparison.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

PY = "/home/nac/anaconda3/envs/llm_train/bin/python"
ROOT = os.path.dirname(os.path.abspath(__file__))

CKPTS = [
    ("qwen",  0, "checkpoints/qwen/seed0/simulator_full.pt",  "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen",  1, "checkpoints/qwen/seed1/simulator_full.pt",  "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen",  2, "checkpoints/qwen/seed2/simulator_full.pt",  "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen",  3, "checkpoints/qwen/seed3/simulator_full.pt",  "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen",  4, "checkpoints/qwen/seed4/simulator_full.pt",  "Qwen/Qwen2.5-1.5B-Instruct"),
    ("llama", 1, "checkpoints/llama/seed1/simulator_full.pt", "unsloth/llama-3-8b-instruct-bnb-4bit"),
    ("llama", 2, "checkpoints/llama/seed2/simulator_full.pt", "unsloth/llama-3-8b-instruct-bnb-4bit"),
    ("llama", 3, "checkpoints/llama/seed3/simulator_full.pt", "unsloth/llama-3-8b-instruct-bnb-4bit"),
    ("llama", 4, "checkpoints/llama/seed4/simulator_full.pt", "unsloth/llama-3-8b-instruct-bnb-4bit"),
]
# Qwen seed 3 is the known bad-init outlier (val_loss 2.28); excluded from trimmed aggregates.
TRIM_EXCLUDE = {("qwen", 3)}

# (eval_tag, items_file, human_file, scale_max, prior_T)
EVALS = [
    ("ipip", "data/item_holdout.parquet", "data/item_holdout.parquet", 5, 0.6),
    ("bfi",  "data/bfi_items.parquet",    "bfi_human_long.csv",        6, 1.0),
]

T_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
N_SILICON = 1000
SPLIT_SEED = 42


def split_humans(human_file, seed=SPLIT_SEED):
    """Deterministic 50/50 split by sorted participant_id."""
    df = pd.read_parquet(human_file) if human_file.endswith(".parquet") else pd.read_csv(human_file)
    pids = np.array(sorted(df.participant_id.unique()))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pids))
    cut = len(pids) // 2
    cal_pids = set(pids[idx[:cut]].tolist())
    rep_pids = set(pids[idx[cut:]].tolist())
    return df[df.participant_id.isin(cal_pids)].copy(), df[df.participant_id.isin(rep_pids)].copy()


def rescale_to_1_5(resp, scale_max):
    """Linearly rescale 1..scale_max → 1..5 so BFI (1-6) is comparable to silicon (1-5)."""
    if scale_max == 5:
        return resp.astype(float)
    return (resp.astype(float) - 1.0) * 4.0 / (scale_max - 1) + 1.0


def calib_objective(human_cal, silicon, scale_max):
    """Per-item moment match (mean+std distance) vs cal-half humans. Lower=better.

    Both means and stds are computed on items present in the silicon CSV.
    Humans are rescaled to 1-5 to match silicon.
    """
    h = human_cal.copy()
    s = silicon.copy()
    h["resp_1to5"] = rescale_to_1_5(h["response"], scale_max)
    s_col = "samp_resp" if "samp_resp" in s.columns else "response"
    h_stats = h.groupby("item_id")["resp_1to5"].agg(["mean", "std"]).rename(
        columns={"mean": "mu_h", "std": "sd_h"})
    s_stats = s.groupby("item_id")[s_col].agg(["mean", "std"]).rename(
        columns={"mean": "mu_s", "std": "sd_s"})
    j = h_stats.join(s_stats, how="inner").dropna()
    mu_rmse = float(np.sqrt(np.mean((j.mu_s - j.mu_h) ** 2)))
    sd_rmse = float(np.sqrt(np.mean((j.sd_s - j.sd_h) ** 2)))
    return mu_rmse + sd_rmse, mu_rmse, sd_rmse


def silicon_sample_at_T(ckpt, model_name, items_file, T, out_csv):
    """Idempotent: skip if out_csv already exists."""
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    cmd = [PY, "silicon_sample.py",
           "--checkpoint", ckpt, "--model_name", model_name,
           "--items_file", items_file, "--temperature", f"{T}",
           "--n", str(N_SILICON), "--seed", "0", "--mode", "full",
           "--output_file", out_csv]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout); sys.stderr.write(r.stderr)
        raise RuntimeError(f"silicon_sample failed at T={T} -> {out_csv}")


PSY_PATTERNS = {
    "tucker":   r"MEAN congruence\s*=\s*([0-9.]+)",
    "phi_frob": r"Frobenius error\s*=\s*([0-9.]+)",
    "itc_corr": r"item-total profile corr.*=\s*([0-9.\-]+)",
}


def psychometrics_against(human_subset_long, silicon_csv):
    """Write subset human file to a temp path, run 05 driver, regex-parse stdout."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    try:
        human_subset_long.to_csv(tmp.name, index=False)
        tmp.close()
        cmd = [PY, "05_simulator_psychometrics.py", "--human", tmp.name, "--silicon", silicon_csv]
        r = subprocess.run(cmd, capture_output=True, text=True)
        out = r.stdout
        res = {}
        for k, p in PSY_PATTERNS.items():
            m = re.search(p, out)
            res[k] = float(m.group(1)) if m else None
        return res
    finally:
        os.unlink(tmp.name)


def _agg_group(df, models, seeds, etag):
    """Mean ± SD Tucker and calib_obj per T for the given (model, seed) pairs."""
    mask = df["eval"] == etag
    for m, s in zip(models, seeds):
        mask = mask & ~((df["model"] == m) & (df["seed"] == s) & False)
    sub = df[df.apply(lambda r: (r["model"], r["seed"]) in set(zip(models, seeds)) and r["eval"] == etag, axis=1)]
    return sub.groupby("T")[["tucker", "calib_obj"]].agg(["mean", "std"])


def write_summary(df, out_md, prior_results):
    """Markdown: aggregate T*/sensitivity tables (mean ± SD across seeds) + per-seed detail."""
    lines = ["# Held-out temperature calibration (§5.6)\n"]
    lines.append("Per (checkpoint, eval), humans are split 50/50 by deterministic permutation "
                 "(seed 42) into a **calibration** half and a **reporting** half. The "
                 "calibration half picks T\\* by minimising a per-item moment-match objective "
                 "(mean+std RMSE between human and silicon response distributions, BFI rescaled "
                 "to 1–5). The reporting half is held out for the headline psychometrics. "
                 "Aggregates use Qwen trimmed n=4 (excluding seed 3, bad-init) and Llama n=4.\n")

    # --- Build per-seed T* map ---
    chosen = {}
    for (model, seed, _, _) in CKPTS:
        for (etag, _, _, _, _) in EVALS:
            sub = df[(df.model == model) & (df.seed == seed) & (df["eval"] == etag)]
            if not sub.empty:
                chosen[(model, seed, etag)] = float(sub.sort_values("calib_obj").iloc[0]["T"])

    # --- Group definitions ---
    qwen_all  = [(m, s) for (m, s, _, _) in CKPTS if m == "qwen"]
    qwen_trim = [(m, s) for (m, s) in qwen_all if (m, s) not in TRIM_EXCLUDE]
    llama_all = [(m, s) for (m, s, _, _) in CKPTS if m == "llama"]
    GROUPS = [
        ("Qwen (n=4 trim)", qwen_trim, "qwen"),
        ("Llama (n=4)",     llama_all, "llama"),
    ]

    def group_sub(df, pairs, etag):
        return df[df.apply(lambda r: (r["model"], r["seed"]) in set(pairs) and r["eval"] == etag, axis=1)]

    # --- Table: aggregate T* ---
    lines.append("## Aggregate calibration-optimal T\\* (mean ± SD across seeds)\n")
    lines.append("| Backbone | Inventory | Prior T | Mean T\\* | SD T\\* | Mean calib obj | SD calib obj |")
    lines.append("|---|---|---|---|---|---|---|")
    for (label, pairs, model_tag) in GROUPS:
        for (etag, _, _, _, prior_T) in EVALS:
            stars = [chosen.get((m, s, etag), np.nan) for (m, s) in pairs]
            objs  = []
            for (m, s) in pairs:
                sub = df[(df.model == m) & (df.seed == s) & (df["eval"] == etag)]
                if not sub.empty:
                    objs.append(float(sub.sort_values("calib_obj").iloc[0]["calib_obj"]))
            stars = [v for v in stars if not np.isnan(v)]
            mean_s = np.mean(stars) if stars else np.nan
            std_s  = np.std(stars, ddof=1) if len(stars) > 1 else 0.0
            mean_o = np.mean(objs) if objs else np.nan
            std_o  = np.std(objs, ddof=1) if len(objs) > 1 else 0.0
            lines.append(f"| {label} | {etag.upper()} | {prior_T} | "
                         f"{mean_s:.2f} | {std_s:.2f} | {mean_o:.3f} | {std_o:.3f} |")

    # --- Table: aggregate Tucker sensitivity ---
    T_SHOW = [0.4, 0.6, 0.8, 1.0, 1.2]
    lines.append("\n## Aggregate Tucker sensitivity (mean ± SD, rep-half humans)\n")
    head = "| Backbone × inventory | " + " | ".join(f"T={t}" for t in T_SHOW) + " | Mean spread |"
    sep  = "|" + "|".join(["---"] * (1 + len(T_SHOW) + 1)) + "|"
    lines.append(head); lines.append(sep)
    for (label, pairs, _) in GROUPS:
        for (etag, _, _, _, _) in EVALS:
            sub = group_sub(df, pairs, etag)
            cells = [f"{label} × {etag.upper()}"]
            tuckers_per_seed = {}
            for (m, s) in pairs:
                ssub = sub[(sub.model == m) & (sub.seed == s)]
                tuckers_per_seed[(m, s)] = {float(r["T"]): r["tucker"] for _, r in ssub.iterrows()}
            spreads = []
            for (m, s), td in tuckers_per_seed.items():
                vals = [v for v in td.values() if pd.notna(v)]
                if vals:
                    spreads.append(max(vals) - min(vals))
            for t in T_SHOW:
                vals = [td.get(t, np.nan) for td in tuckers_per_seed.values()]
                vals = [v for v in vals if pd.notna(v)]
                if vals:
                    cells.append(f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}")
                else:
                    cells.append("n/a")
            cells.append(f"{np.mean(spreads):.3f} ± {np.std(spreads, ddof=1):.3f}" if len(spreads) > 1 else f"{np.mean(spreads):.3f}")
            lines.append("| " + " | ".join(cells) + " |")

    # --- Per-seed detail ---
    lines.append("\n## Per-seed detail\n")
    lines.append("| ckpt | eval | T\\* | calib obj | Tucker @ prior T | Tucker @ T\\* | Δ |")
    lines.append("|---|---|---|---|---|---|---|")
    for (model, seed, _, _) in CKPTS:
        for (etag, _, _, _, prior_T) in EVALS:
            sub = df[(df.model == model) & (df.seed == seed) & (df["eval"] == etag)]
            if sub.empty: continue
            star_T = chosen.get((model, seed, etag), prior_T)
            prior_row = sub[np.isclose(sub["T"], prior_T)]
            star_row  = sub[np.isclose(sub["T"], star_T)]
            if prior_row.empty or star_row.empty: continue
            tp = prior_row["tucker"].iloc[0]; ts = star_row["tucker"].iloc[0]
            obj = star_row["calib_obj"].iloc[0]
            lines.append(f"| {model}/seed{seed} | {etag} | {star_T:.2f} | {obj:.3f} | "
                         f"{tp:.3f} (T={prior_T}) | {ts:.3f} | {ts - tp:+.3f} |")

    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {out_md}")


def main(args):
    os.chdir(ROOT)
    os.makedirs("results", exist_ok=True)

    # Per-checkpoint headline Tucker from §5.5 (used in per-seed detail table).
    PUBLISHED = {
        ("qwen",  0, "ipip"): 0.842, ("qwen",  0, "bfi"): 0.858,
        ("qwen",  1, "ipip"): 0.864, ("qwen",  1, "bfi"): 0.886,
        ("qwen",  2, "ipip"): 0.834, ("qwen",  2, "bfi"): 0.854,
        ("qwen",  3, "ipip"): 0.712, ("qwen",  3, "bfi"): 0.658,
        ("qwen",  4, "ipip"): 0.858, ("qwen",  4, "bfi"): 0.886,
        ("llama", 1, "ipip"): 0.908, ("llama", 1, "bfi"): 0.884,
        ("llama", 2, "ipip"): 0.834, ("llama", 2, "bfi"): 0.860,
        ("llama", 3, "ipip"): 0.894, ("llama", 3, "bfi"): 0.874,
        ("llama", 4, "ipip"): 0.854, ("llama", 4, "bfi"): 0.752,
    }

    rows = []
    t_total = time.time()
    for (model, seed, ckpt, model_name) in CKPTS:
        for (etag, items, human_path, scale_max, prior_T) in EVALS:
            print(f"\n=== {model}/seed{seed} :: {etag} ===")
            human_cal, human_rep = split_humans(human_path)
            print(f"  cal humans: {human_cal.participant_id.nunique()}  "
                  f"rep humans: {human_rep.participant_id.nunique()}")
            for T in T_GRID:
                out_csv = f"silicon_outputs/{model}/seed{seed}/calib_T{T:.2f}_{etag}.csv"
                print(f"  T={T:.2f}: ", end="", flush=True)
                t0 = time.time()
                silicon_sample_at_T(ckpt, model_name, items, T, out_csv)
                print(f"silicon ({time.time()-t0:.0f}s) ", end="", flush=True)

                silicon = pd.read_csv(out_csv)
                obj, mu_rmse, sd_rmse = calib_objective(human_cal, silicon, scale_max)
                psy = psychometrics_against(human_rep, out_csv)
                print(f"obj={obj:.3f} tucker={psy['tucker']}")

                rows.append({
                    "model": model, "seed": seed, "eval": etag, "T": T,
                    "calib_obj": obj, "mu_rmse": mu_rmse, "sd_rmse": sd_rmse,
                    "tucker": psy["tucker"], "phi_frob": psy["phi_frob"],
                    "itc_corr": psy["itc_corr"],
                })
    df = pd.DataFrame(rows)
    df.sort_values(["model", "seed", "eval", "T"], inplace=True)
    df.to_csv("results/temp_calibration.csv", index=False)
    print(f"\nWrote results/temp_calibration.csv  ({len(df)} rows; "
          f"total {time.time()-t_total:.0f}s)")

    # Aggregate CSV: mean ± SD per (model, eval, T), all seeds and trimmed.
    agg_rows = []
    for (model_tag, label, pairs) in [
        ("qwen_all",  "Qwen all (n=5)",   [(m, s) for (m, s, _, _) in CKPTS if m == "qwen"]),
        ("qwen_trim", "Qwen trim (n=4)",  [(m, s) for (m, s, _, _) in CKPTS if m == "qwen" and (m, s) not in TRIM_EXCLUDE]),
        ("llama",     "Llama (n=4)",      [(m, s) for (m, s, _, _) in CKPTS if m == "llama"]),
    ]:
        for (etag, _, _, _, prior_T) in EVALS:
            sub = df[df.apply(lambda r: (r["model"], r["seed"]) in set(pairs) and r["eval"] == etag, axis=1)]
            for T_val in T_GRID:
                tsub = sub[np.isclose(sub["T"], T_val)]
                agg_rows.append({
                    "group": model_tag, "label": label, "eval": etag, "prior_T": prior_T, "T": T_val,
                    "n": len(tsub),
                    "tucker_mean": tsub["tucker"].mean(), "tucker_std": tsub["tucker"].std(ddof=1),
                    "calib_obj_mean": tsub["calib_obj"].mean(), "calib_obj_std": tsub["calib_obj"].std(ddof=1),
                })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv("results/temp_calibration_aggregate.csv", index=False)
    print(f"Wrote results/temp_calibration_aggregate.csv  ({len(agg_df)} rows)")

    write_summary(df, "results/temp_calibration.md", PUBLISHED)
    print("\n" + open("results/temp_calibration.md").read())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--noop", action="store_true", help="(placeholder)")
    main(p.parse_args())
