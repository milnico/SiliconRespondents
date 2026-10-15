"""Pipeline validation via a genuine human split-half.

The psychometrics pipeline (05_simulator_psychometrics.R) is validated by
splitting the *human* sample into two random, disjoint halves and running the
exact same factor-analysis machinery between them. This establishes an honest
human **noise ceiling**: the best Tucker congruence / smallest Phi error any
silicon population could realistically match, given ordinary sampling
variability between two subsamples of the same population.

This supersedes the earlier "identity" sanity check (the same file compared with
itself), which trivially returns congruence 1.000 / Phi error 0 / identical
alphas -- a tautology, not an empirical split-half. A real split of independent
respondents gives congruence < 1 and a strictly positive Phi error.

Usage:
  python 11_pipeline_validation.py --human data/item_holdout.parquet
  python 11_pipeline_validation.py --human bfi_human_long.csv --out results/pipeline_validation.md

Metrics are correlation/covariance based, so the IPIP (1-5) vs BFI (1-6) scale
difference is irrelevant for a human-vs-human comparison (no rescaling needed).
"""
import argparse
import os
import subprocess
import tempfile
import numpy as np
import pandas as pd

RSCRIPT_DEFAULT = "/home/nac/anaconda3/envs/llm_train/bin/Rscript"
SPLIT_SEED = 42


def load_long(path):
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def split_humans(df, seed=SPLIT_SEED):
    """Deterministic 50/50 split of participants into two disjoint halves.

    Mirrors the splitter pattern in 08_temp_calibration.py: permute the sorted
    unique participant_ids with a seeded RNG and cut in the middle.
    """
    pids = np.array(sorted(df.participant_id.unique()))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pids))
    cut = len(pids) // 2
    a_pids = set(pids[idx[:cut]].tolist())
    b_pids = set(pids[idx[cut:]].tolist())
    return df[df.participant_id.isin(a_pids)].copy(), df[df.participant_id.isin(b_pids)].copy()


def normalise(df, meta):
    """Long frame with the schema 05_simulator_psychometrics.R expects:
    participant_id,item_id,factor,reverse,resp."""
    out = df[["participant_id", "item_id", "response"]].rename(columns={"response": "resp"})
    out = out.merge(meta, on="item_id", how="left")
    return out[["participant_id", "item_id", "factor", "reverse", "resp"]]


def main(args):
    human = load_long(args.human)
    if "response" not in human.columns:
        raise ValueError(f"human file must contain a 'response' column; got {list(human.columns)}")

    # Item metadata (factor, reverse) is authoritative from the full sample.
    meta = human.drop_duplicates("item_id")[["item_id", "factor", "reverse"]]

    half_a, half_b = split_humans(human, seed=args.seed)
    n_a = half_a.participant_id.nunique()
    n_b = half_b.participant_id.nunique()
    overlap = set(half_a.participant_id) & set(half_b.participant_id)
    assert not overlap, "split halves overlap -- partition is broken"
    print(f"Human split-half on {args.human}: half A = {n_a} ppl, half B = {n_b} ppl, "
          f"overlap = {len(overlap)} (seed {args.seed})\n")

    tmp = tempfile.mkdtemp(prefix="pipeline_val_")
    af, bf = os.path.join(tmp, "half_a.csv"), os.path.join(tmp, "half_b.csv")
    normalise(half_a, meta).to_csv(af, index=False)
    normalise(half_b, meta).to_csv(bf, index=False)

    cmd = [args.rscript, "05_simulator_psychometrics.R", af, bf]
    print(f"Running: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, check=True, capture_output=bool(args.out), text=True)

    if args.out:
        header = (f"# Pipeline validation — human split-half ({os.path.basename(args.human)})\n\n"
                  f"Two disjoint random halves of the human sample (seed {args.seed}): "
                  f"half A = {n_a} ppl, half B = {n_b} ppl.\n\n"
                  f"```\n{proc.stdout}```\n")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(header)
        print(proc.stdout)
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--human", default="data/item_holdout.parquet",
                   help="human long data (parquet/csv) with a 'response' column")
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--rscript", default=RSCRIPT_DEFAULT)
    p.add_argument("--out", default=None, help="optional path to write R output as markdown")
    main(p.parse_args())
