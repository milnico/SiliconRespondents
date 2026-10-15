"""Multi-seed / cross-backbone evaluation sweep.

Walks every checkpoint produced by `train_campaign.sh` (Qwen seeds 0-4 +
Llama-4bit seeds 1-2), generates three silicon samples per checkpoint
(IPIP held-out, BFI OOD, cross-domain criterion battery), and computes:
  * Structural psychometrics (Tucker mean congruence, Frobenius Phi error,
    silicon mean corrected item-total correlation, and its MAE vs human) --
    via 05_simulator_psychometrics.py (R).
  * Criterion validity (per-domain profile corr vs literature, mean profile
    corr, matrix RMSE) -- by reusing helpers from 06_criterion_validity.py.

Outputs results/per_checkpoint.csv (long format) and results/summary.md
(per-backbone mean +/- sd, with an n=4 trimmed-mean Qwen row that drops the
seed-3 outlier; cross-backbone Qwen-vs-Llama delta).

Idempotent: skips a silicon CSV if it already exists, so Ctrl-C / re-run is
safe. Per-checkpoint wall-clock ~3.5 min on Qwen, ~10 min on Llama-4bit.
"""
import argparse
import importlib.util
import os
import re
import subprocess
import sys
import time
from glob import glob

import numpy as np
import pandas as pd

PY = "/home/nac/anaconda3/envs/llm_train/bin/python"
ROOT = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    # tag -> HF model name passed as --model_name
    "qwen":  "Qwen/Qwen2.5-1.5B-Instruct",
    "llama": "unsloth/llama-3-8b-instruct-bnb-4bit",
}

# (test_set tag, items_file, temperature, n, human_file [for 05])
SAMPLES = [
    ("ipip",      "data/item_holdout.parquet",      0.6, 500,  "data/item_holdout.parquet"),
    ("bfi",       "data/bfi_items.parquet",         1.0, 500,  "bfi_human_long.csv"),
    ("criterion", "data/criterion_battery.parquet", 1.0, 1000, None),
]

TRAITS = ["O", "C", "E", "A", "N"]


# --------------------------------------------------------------------------- #
# Import the existing criterion-validity helpers (file starts with a digit so
# we can't use `import 06_criterion_validity` directly).
# --------------------------------------------------------------------------- #
def _load_06():
    spec = importlib.util.spec_from_file_location("crit", os.path.join(ROOT, "06_criterion_validity.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

crit = _load_06()


# --------------------------------------------------------------------------- #
# Silicon sampling
# --------------------------------------------------------------------------- #
def silicon_sample(checkpoint, model_name, items_file, temperature, n, output_file):
    """Run silicon_sample.py if `output_file` doesn't already exist."""
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"  [skip] {output_file} exists")
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cmd = [PY, "silicon_sample.py",
           "--checkpoint", checkpoint, "--model_name", model_name,
           "--items_file", items_file, "--temperature", str(temperature),
           "--n", str(n), "--seed", "0",
           "--output_file", output_file, "--mode", "full"]
    print(f"  silicon_sample -> {output_file}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout); sys.stderr.write(r.stderr)
        raise RuntimeError(f"silicon_sample failed for {output_file}")
    print(f"    done ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- #
# Structural psychometrics (parse 05_simulator_psychometrics stdout, same regex
# pattern as aggregate_results.py:30-38).
# --------------------------------------------------------------------------- #
PSY_PATTERNS = {
    "congruence":     r"MEAN congruence\s*=\s*([0-9.]+)",
    "per_factor":     r"per factor:\s*([0-9. ]+)",
    "phi_frobenius":  r"Frobenius error\s*=\s*([0-9.]+)",
    # Silicon mean corrected item-total correlation over the held-out items.
    "itemtotal_r":    r"mean corrected item-total \(over items\):.*?silicon=\s*([0-9.\-]+)",
    # Mean |silicon - human| corrected item-total correlation (agreement; lower=better).
    # NOTE non-greedy `.*?` so this grabs the MAE field, not the profile-corr field.
    "itemtotal_mae":  r"item-total profile corr.*?\|\s*MAE\s*=\s*([0-9.\-]+)",
    # Pearson corr between the silicon and human per-item item-total vectors.
    "itemtotal_profilecorr": r"item-total profile corr.*?=\s*([0-9.\-]+)",
}


def run_psychometrics(human_file, silicon_csv):
    cmd = [PY, "05_simulator_psychometrics.py", "--human", human_file, "--silicon", silicon_csv]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout
    res = {}
    for key, pat in PSY_PATTERNS.items():
        m = re.search(pat, out)
        if not m:
            res[key] = None
            continue
        val = m.group(1).strip()
        res[key] = val if key == "per_factor" else float(val)
    return res


# --------------------------------------------------------------------------- #
# Criterion validity (numeric — capture per-domain profile corr, mean, RMSE,
# dominant-trait/sign match, dominant-r bootstrap CI).
# --------------------------------------------------------------------------- #
def run_criterion(silicon_csv, meta, lit, n_boot=2000, seed=0):
    sil = pd.read_csv(silicon_csv)
    scores = crit.scale_scores(sil, meta)
    domains = lit["scale"].tolist()
    target = (meta[meta.scale_type == "domain"]
              .drop_duplicates("scale").set_index("scale")[["target_trait", "target_sign"]])

    rows, all_obs, all_lit = [], [], []
    for dom in domains:
        litvec = lit.set_index("scale").loc[dom, TRAITS].to_numpy(dtype=float)
        obs = np.array([crit.corr(scores[dom], scores[t]) for t in TRAITS])
        all_obs.append(obs); all_lit.append(litvec)

        ttrait, tsign = target.loc[dom, "target_trait"], int(target.loc[dom, "target_sign"])
        ti = TRAITS.index(ttrait)
        argmax_i = int(np.nanargmax(np.abs(obs)))
        sign_ok = np.sign(obs[ti]) == tsign
        dominant_ok = (argmax_i == ti) and bool(sign_ok)
        lo, hi = crit.boot_ci(scores[dom].to_numpy(), scores[ttrait].to_numpy(), n_boot, seed)

        rows.append({
            "domain": dom, "target_trait": ttrait, "target_sign": tsign,
            "argmax_trait": TRAITS[argmax_i], "dominant_match": dominant_ok,
            "sign_match_on_target": bool(sign_ok),
            "r_target": float(obs[ti]), "ci_lo_target": float(lo), "ci_hi_target": float(hi),
            "profile_corr_vs_lit": float(crit.profile_corr(obs, litvec)),
            **{f"r_{t}": float(o) for t, o in zip(TRAITS, obs)},
        })

    rmse = float(np.sqrt(np.nanmean((np.array(all_obs) - np.array(all_lit)) ** 2)))
    mean_pc = float(np.nanmean([crit.profile_corr(o, l) for o, l in zip(all_obs, all_lit)]))
    return {"per_domain": rows, "rmse": rmse, "mean_profile_corr": mean_pc}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def discover_checkpoints():
    out = []
    for tag in ["qwen", "llama"]:
        for ckpt in sorted(glob(os.path.join(ROOT, f"checkpoints/{tag}/seed*/simulator_full.pt"))):
            seed = int(re.search(r"seed(\d+)", ckpt).group(1))
            out.append((tag, seed, ckpt))
    return out


def fmt_mean_sd(vals, fmt="{:.3f}"):
    a = np.array([v for v in vals if v is not None and not np.isnan(v)])
    if len(a) == 0:
        return "n/a"
    if len(a) == 1:
        return f"{fmt.format(a[0])} (n=1)"
    return f"{fmt.format(a.mean())} ± {fmt.format(a.std(ddof=1))}"


def write_summary(df, out_md):
    """Build a markdown summary with mean ± sd per backbone, trimmed Qwen row,
    and Qwen-vs-Llama delta on headline metrics."""
    metrics = [
        ("ipip_congruence",         "IPIP Tucker"),
        ("ipip_phi_frobenius",      "IPIP Φ err"),
        ("ipip_itemtotal_r",        "IPIP item-tot r"),
        ("bfi_congruence",          "BFI Tucker"),
        ("bfi_phi_frobenius",       "BFI Φ err"),
        ("bfi_itemtotal_r",         "BFI item-tot r"),
        ("criterion_mean_pc",       "Crit mean prof.corr"),
        ("criterion_rmse",          "Crit RMSE"),
    ]

    lines = []
    lines.append("# Multi-seed / cross-backbone evaluation\n")
    lines.append("## Per-checkpoint\n")
    head = "| model | seed | best val_loss | " + " | ".join(m[1] for m in metrics) + " |"
    sep  = "|" + "|".join(["---"] * (3 + len(metrics))) + "|"
    lines.append(head); lines.append(sep)
    for _, r in df.iterrows():
        cells = [r["model"], str(int(r["seed"])), f"{r['val_loss']:.3f}" if r["val_loss"] is not None else "—"]
        for k, _ in metrics:
            v = r.get(k)
            cells.append(f"{v:.3f}" if isinstance(v, (int, float)) and not np.isnan(v) else "n/a")
        lines.append("| " + " | ".join(cells) + " |")

    qwen = df[df.model == "qwen"]
    qwen_trim = qwen[qwen.seed != 3]
    llama = df[df.model == "llama"]

    lines.append("\n## Aggregated (mean ± sd)\n")
    head = "| group | " + " | ".join(m[1] for m in metrics) + " |"
    sep  = "|" + "|".join(["---"] * (1 + len(metrics))) + "|"
    lines.append(head); lines.append(sep)

    for label, sub in [
        (f"Qwen (n={len(qwen)})",            qwen),
        (f"Qwen trimmed (n={len(qwen_trim)}, drop seed3)", qwen_trim),
        (f"Llama-4bit (n={len(llama)})",     llama),
    ]:
        cells = [label]
        for k, _ in metrics:
            cells.append(fmt_mean_sd(sub[k].tolist()))
        lines.append("| " + " | ".join(cells) + " |")

    # Cross-backbone delta on headline metrics (using trimmed Qwen for fairness vs the bad init).
    lines.append("\n## Cross-backbone Δ (Qwen trimmed − Llama)\n")
    head = "| metric | Qwen trimmed | Llama | Δ |"
    lines.append(head); lines.append("|---|---|---|---|")
    for k, name in metrics:
        q = qwen_trim[k].astype(float).mean()
        l = llama[k].astype(float).mean()
        d = q - l
        lines.append(f"| {name} | {q:.3f} | {l:.3f} | {d:+.3f} |")

    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {out_md}")


def main(args):
    os.chdir(ROOT)
    os.makedirs("results", exist_ok=True)

    meta = pd.read_csv("data/criterion_meta.csv")
    lit = pd.read_csv("data/criterion_literature.csv")

    # Per-run best val loss (parsed from training logs once).
    val_losses = {}
    for log in glob("logs/train_*_seed*.log"):
        m = re.search(r"train_(\w+)_seed(\d+)\.log", log)
        if not m: continue
        tag, seed = m.group(1), int(m.group(2))
        # The "new best -> saved" lines are immediately preceded by a val-loss line.
        with open(log) as f:
            txt = f.read()
        vals = re.findall(r"item-holdout loss=([0-9.]+)", txt)
        val_losses[(tag, seed)] = float(min(vals)) if vals else None
    # seed-0 (the copied original Qwen run) — pull from CLAUDE.md note (1.336).
    val_losses.setdefault(("qwen", 0), 1.336)

    rows = []
    per_domain_rows = []
    ckpts = discover_checkpoints()
    print(f"Discovered {len(ckpts)} checkpoints.")
    for tag, seed, ckpt in ckpts:
        print(f"\n=== {tag} seed{seed} ===  ({ckpt})")
        out_dir = os.path.join(ROOT, "silicon_outputs", tag, f"seed{seed}")
        os.makedirs(out_dir, exist_ok=True)

        # 1. Sample
        sil_paths = {}
        for s_tag, items, temp, n, _ in SAMPLES:
            out_csv = os.path.join(out_dir, f"{s_tag}.csv")
            silicon_sample(ckpt, MODELS[tag], items, temp, n, out_csv)
            sil_paths[s_tag] = out_csv

        row = {"model": tag, "seed": seed, "val_loss": val_losses.get((tag, seed))}

        # 2. Structural psychometrics (IPIP + BFI)
        for s_tag, items, temp, n, human in SAMPLES:
            if human is None:
                continue
            print(f"  psychometrics [{s_tag}] ...")
            res = run_psychometrics(human, sil_paths[s_tag])
            row[f"{s_tag}_congruence"]      = res["congruence"]
            row[f"{s_tag}_phi_frobenius"]   = res["phi_frobenius"]
            row[f"{s_tag}_itemtotal_r"]     = res["itemtotal_r"]
            row[f"{s_tag}_itemtotal_mae"]   = res["itemtotal_mae"]
            row[f"{s_tag}_per_factor"]      = res["per_factor"]

        # 3. Criterion validity
        print(f"  criterion validity ...")
        crit_res = run_criterion(sil_paths["criterion"], meta, lit, n_boot=args.n_boot, seed=0)
        row["criterion_mean_pc"] = crit_res["mean_profile_corr"]
        row["criterion_rmse"]    = crit_res["rmse"]
        for d in crit_res["per_domain"]:
            per_domain_rows.append({"model": tag, "seed": seed, **d})

        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values(["model", "seed"], inplace=True)
    df.to_csv("results/per_checkpoint.csv", index=False)
    pd.DataFrame(per_domain_rows).to_csv("results/per_checkpoint_criterion_detail.csv", index=False)
    print(f"\nWrote results/per_checkpoint.csv  ({len(df)} rows)")
    print(f"Wrote results/per_checkpoint_criterion_detail.csv  ({len(per_domain_rows)} rows)")

    write_summary(df, "results/summary.md")
    # Also dump the summary to stdout for the live report.
    print("\n" + open("results/summary.md").read())

    # Discriminant-validity β analysis (writes results/betas_*.{csv,md}).
    print("\n>>> running 07_partial_betas.py ...")
    subprocess.run([PY, "07_partial_betas.py", "--n_boot", str(args.n_boot)], check=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_boot", type=int, default=2000)
    main(p.parse_args())
