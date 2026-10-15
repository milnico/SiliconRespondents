"""Driver for the R psychometric evaluation.

Normalises the human and silicon long files to a common CSV schema
(participant_id,item_id,factor,reverse,resp) and shells out to
05_simulator_psychometrics.R (psych/GPArotation) for the factor analysis,
since R is preferred for this kind of analysis (and lacks `arrow`, so parquet
must be converted here).
"""
import argparse
import os
import subprocess
import tempfile
import pandas as pd

RSCRIPT_DEFAULT = "/home/nac/anaconda3/envs/llm_train/bin/Rscript"


def load_long(path):
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def pick_resp(df, prefer):
    for c in [prefer, "samp_resp", "response", "pred_resp", "true_resp"]:
        if c in df.columns:
            return c
    raise ValueError(f"No response column found in {list(df.columns)}")


def normalise(df, resp_col, meta):
    out = df[["participant_id", "item_id", resp_col]].rename(columns={resp_col: "resp"})
    out = out.merge(meta, on="item_id", how="left")
    return out[["participant_id", "item_id", "factor", "reverse", "resp"]]


def main(args):
    human = load_long(args.human)
    silicon = load_long(args.silicon)

    # Item metadata (factor, reverse) comes from the human file (authoritative).
    if "reverse" not in human.columns:
        raise ValueError("human file must contain a 'reverse' column")
    meta = human.drop_duplicates("item_id")[["item_id", "factor", "reverse"]]

    h = normalise(human, pick_resp(human, "response"), meta)
    s = normalise(silicon, pick_resp(silicon, "samp_resp"), meta)

    tmp = tempfile.mkdtemp(prefix="psychometrics_")
    hf, sf = os.path.join(tmp, "human_norm.csv"), os.path.join(tmp, "silicon_norm.csv")
    h.to_csv(hf, index=False)
    s.to_csv(sf, index=False)

    cmd = [args.rscript, "05_simulator_psychometrics.R", hf, sf]
    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--human", required=True, help="human long data (parquet/csv) with 'response'")
    p.add_argument("--silicon", required=True, help="silicon sample csv with 'samp_resp'")
    p.add_argument("--rscript", default=RSCRIPT_DEFAULT)
    args = p.parse_args()
    main(args)
