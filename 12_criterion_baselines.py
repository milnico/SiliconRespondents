"""Baselines for the criterion profile-correlation metric (§5.4).

The §5.4 nomological-validity claim rests on the Pearson "profile correlation"
between the silicon domain->trait r-vector and a meta-analytic literature target
vector. That metric has limited discriminating power, because the literature
targets are themselves nearly evaluative-halo shaped. This script computes two
baselines against the SAME literature targets so the model can be read against
them honestly:

  * HALO       -- a pure evaluative-halo vector h=(+,+,+,+,-) on O/C/E/A/N,
                  oriented a priori by domain valence (equivalently |corr(h,lit)|).
                  Needs no model at all.
  * SEMANTIC   -- predicts each domain's trait profile from ITEM TEXT alone:
                  embed every battery item with the frozen backbone, build
                  sign-aware keyed direction vectors per Big Five trait (from the
                  20 marker items) and per domain (from its items), and take the
                  cosine between them. Tests whether the "emergent" links are just
                  recoverable from item wording by any LLM.
  * MODEL      -- the silicon zero-order profile corr (from an existing criterion
                  CSV), for side-by-side comparison.
  * FLOOR      -- the no-trait baseline_a criterion CSV, if available (expected ~0).

Outputs results/criterion_baselines.{csv,md}. No GPU sampling; embeddings use the
already-cached backbone. Invoke the interpreter bare with the conda env on PATH.
"""
import argparse
import numpy as np
import pandas as pd

TRAITS = ["O", "C", "E", "A", "N"]
SCALE_MAX = 5
# High score = desirable? (standard evaluative direction: high O/C/E/A good, high N bad).
DESIRABLE = {"GRIT": True, "SELF": True, "LEAD": True, "PROC": False}


def profile_corr(vec, lit):
    vec = np.asarray(vec, float)
    if np.std(vec) == 0:
        return np.nan
    return float(np.corrcoef(vec, lit)[0, 1])


def silicon_profile(criterion_csv, meta, lit):
    """Model zero-order r-vector per domain from a criterion sample CSV."""
    df = pd.read_csv(criterion_csv)
    m = df.merge(meta[["item_id", "scale", "reverse"]], on="item_id", how="inner")
    keyed = m["samp_resp"].to_numpy(float)
    rev = m["reverse"].astype(bool).to_numpy()
    keyed[rev] = (SCALE_MAX + 1) - keyed[rev]
    m = m.assign(keyed=keyed)
    wide = m.groupby(["participant_id", "scale"])["keyed"].mean().unstack("scale")
    out = {}
    for dom in lit.index:
        if dom not in wide:
            continue
        vec = [wide[[dom, t]].corr().iloc[0, 1] for t in TRAITS]
        out[dom] = profile_corr(vec, lit.loc[dom, TRAITS].to_numpy(float))
    return out


def halo_profile(lit):
    """Pure evaluative-halo vector, oriented a priori by domain valence."""
    d = np.array([1.0, 1.0, 1.0, 1.0, -1.0])  # +O +C +E +A -N
    out = {}
    for dom in lit.index:
        h = d if DESIRABLE.get(dom, True) else -d
        out[dom] = profile_corr(h, lit.loc[dom, TRAITS].to_numpy(float))
    return out


def embed_items(texts, model_name):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    embs = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=64).to(dev)
            h = model(**enc).last_hidden_state[0]           # (seq, hid)
            mask = enc["attention_mask"][0].unsqueeze(-1).float()
            v = (h * mask).sum(0) / mask.sum()              # mean-pool
            v = v / (v.norm() + 1e-8)
            embs.append(v.float().cpu().numpy())
    return np.vstack(embs)


def semantic_profile(battery, meta, lit, model_name):
    """Predict domain->trait profiles from item text alone (sign-aware)."""
    b = battery.merge(meta[["item_id", "scale", "scale_type", "reverse"]],
                      on="item_id", how="inner").drop_duplicates("item_id")
    E = embed_items(b["item_text"].tolist(), model_name)
    # Remove the common (anisotropy) component that inflates decoder-embedding
    # cosines, then renormalize, before sign-keying.
    E = E - E.mean(0, keepdims=True)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    sign = np.where(b["reverse"].astype(bool).to_numpy(), -1.0, 1.0)[:, None]
    Ek = E * sign                                           # key toward "high scale"

    def direction(mask):
        v = Ek[mask.to_numpy()].mean(0)
        return v / (np.linalg.norm(v) + 1e-8)

    trait_dir = {t: direction(b.scale.eq(t) & b.scale_type.eq("bigfive")) for t in TRAITS}
    out = {}
    for dom in lit.index:
        dd = direction(b.scale.eq(dom) & b.scale_type.eq("domain"))
        vec = [float(dd @ trait_dir[t]) for t in TRAITS]
        out[dom] = (profile_corr(vec, lit.loc[dom, TRAITS].to_numpy(float)), vec)
    return out


def main(a):
    meta = pd.read_csv(a.meta)
    lit = pd.read_csv(a.literature).set_index("scale")
    battery = pd.read_parquet(a.battery)

    halo = halo_profile(lit)
    model = silicon_profile(a.model_csv, meta, lit)
    floor = silicon_profile(a.floor_csv, meta, lit) if a.floor_csv else {}
    sem = semantic_profile(battery, meta, lit, a.model_name)

    doms = list(lit.index)
    rows = []
    for dom in doms:
        rows.append({
            "domain": dom,
            "model": round(model.get(dom, np.nan), 3),
            "halo": round(halo[dom], 3),
            "semantic": round(sem[dom][0], 3),
            "floor": round(floor.get(dom, np.nan), 3) if floor else np.nan,
        })
    res = pd.DataFrame(rows)
    means = {"domain": "MEAN", "model": round(res.model.mean(), 3),
             "halo": round(res.halo.mean(), 3), "semantic": round(res.semantic.mean(), 3),
             "floor": round(res.floor.mean(), 3) if floor else np.nan}
    res = pd.concat([res, pd.DataFrame([means])], ignore_index=True)
    res.to_csv("results/criterion_baselines.csv", index=False)

    with open("results/criterion_baselines.md", "w") as f:
        f.write("# Criterion profile-correlation baselines (§5.4)\n\n")
        f.write("Profile r = Pearson corr between each source's domain->trait vector "
                "and the meta-analytic literature target.\n\n")
        cols = list(res.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for _, r in res.iterrows():
            f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
        f.write("\n## Semantic predicted profiles (O C E A N)\n\n")
        for dom in doms:
            v = ", ".join(f"{x:+.2f}" for x in sem[dom][1])
            f.write(f"- {dom}: [{v}]  (profile r = {sem[dom][0]:.3f})\n")
    print(res.to_string(index=False))
    print("\nwrote results/criterion_baselines.{csv,md}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--battery", default="data/criterion_battery.parquet")
    p.add_argument("--meta", default="data/criterion_meta.csv")
    p.add_argument("--literature", default="data/criterion_literature.csv")
    p.add_argument("--model_csv", default="silicon_outputs/qwen/seed0/criterion.csv")
    p.add_argument("--floor_csv", default="silicon_criterion_floor.csv")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    main(p.parse_args())
