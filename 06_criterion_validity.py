"""Cross-domain CRITERION VALIDITY of the silicon population.

Tests whether silicon respondents reproduce *known* personality<->domain
correlations from the literature for instruments the model never trained on
(grit, self-esteem, leadership, procrastination). The domain<->trait link is
NOT in the injected latent covariance, so reproducing it is evidence the
synthetic sample is a true silicon population, not a replay of injected
structure.

For each domain scale we form two correlation vectors against the Big Five:
  * latent-based   -- domain score vs the injected latent z (mechanistic check)
  * observed-based -- domain score vs the silicon's OWN Big Five marker scores
                      (matches how human studies are run; the headline)
and compare to meta-analytic targets (data/criterion_literature.csv):
  * dominant trait + sign match  (discriminant validity)
  * profile correlation across the 5 traits
  * matrix RMSE
  * bootstrap CI on the dominant-trait correlation
A no-trait floor (baseline_a) is contrasted to show the signal comes from
trait conditioning.

Usage:
  python 06_criterion_validity.py --silicon silicon_criterion_full.csv \
      --floor silicon_criterion_floor.csv \
      --meta data/criterion_meta.csv --literature data/criterion_literature.csv
"""
import argparse
import numpy as np
import pandas as pd

TRAITS = ["O", "C", "E", "A", "N"]
SCALE_MAX = 5  # silicon answers on the 1-5 digit scale regardless of instrument


def scale_scores(df, meta):
    """Per-respondent mean score for every scale (reverse-keyed), plus the
    injected latent z. Returns a wide frame indexed by participant_id with
    columns: O,C,E,A,N (observed markers), the domain scales, and Oz..Nz."""
    resp_col = next(c for c in ["samp_resp", "response", "pred_resp", "true_resp"]
                    if c in df.columns)
    m = df.merge(meta[["item_id", "scale", "reverse"]], on="item_id", how="inner")
    keyed = m[resp_col].to_numpy(dtype=float)
    rev = m["reverse"].astype(bool).to_numpy()
    keyed[rev] = (SCALE_MAX + 1) - keyed[rev]
    m = m.assign(keyed=keyed)
    wide = m.groupby(["participant_id", "scale"])["keyed"].mean().unstack("scale")
    latent = (df.groupby("participant_id")[TRAITS].first()
              .rename(columns={t: t + "z" for t in TRAITS}))
    return wide.join(latent)


def corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def profile_corr(vec, lit):
    if np.std(vec) == 0:
        return np.nan
    return np.corrcoef(vec, lit)[0, 1]


def boot_ci(x, y, n_boot, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    vals = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        vals.append(corr(x[s], y[s]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return lo, hi


def _zscore(M):
    """Column-wise z-score; columns with zero variance return zeros (guarded)."""
    sd = M.std(axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (M - M.mean(axis=0)) / sd


def standardized_betas(X, y):
    """Multivariate regression `y ~ X` on z-scored inputs/outputs.

    Returns (beta, r2) where `beta` is the standardized regression coefficient
    vector (length = X.shape[1]) — directly comparable to a partial correlation
    when the X columns are weakly intercorrelated. Uses np.linalg.lstsq (no
    statsmodels dependency).
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    if y.std() == 0 or np.any(X.std(axis=0) == 0):
        return np.full(X.shape[1], np.nan), np.nan
    Xz = _zscore(X); yz = _zscore(y.reshape(-1, 1)).ravel()
    beta, *_ = np.linalg.lstsq(Xz, yz, rcond=None)
    pred = Xz @ beta
    ss_res = float(np.sum((yz - pred) ** 2))
    ss_tot = float(np.sum(yz ** 2))  # already centered
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, r2


def betas_with_ci(X, y, n_boot, seed):
    """Bootstrap CIs for each standardized β by resampling respondents."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.full((n_boot, X.shape[1]), np.nan)
    for i in range(n_boot):
        s = rng.choice(n, n, replace=True)
        b, _ = standardized_betas(X[s], y[s])
        boots[i] = b
    point, r2 = standardized_betas(X, y)
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return point, lo, hi, r2


def analyze(scores, meta, lit, n_boot, seed, label):
    domains = lit["scale"].tolist()
    target = (meta[meta.scale_type == "domain"]
              .drop_duplicates("scale").set_index("scale")[["target_trait", "target_sign"]])

    print(f"\n{'='*78}\n{label}\n{'='*78}")
    all_obs, all_lit = [], []
    for dom in domains:
        litvec = lit.set_index("scale").loc[dom, TRAITS].to_numpy(dtype=float)
        obs = np.array([corr(scores[dom], scores[t]) for t in TRAITS])
        lat = np.array([corr(scores[dom], scores[t + "z"]) for t in TRAITS])
        all_obs.append(obs)
        all_lit.append(litvec)

        ttrait, tsign = target.loc[dom, "target_trait"], int(target.loc[dom, "target_sign"])
        ti = TRAITS.index(ttrait)
        dom_i = int(np.nanargmax(np.abs(obs)))  # silicon's strongest correlate
        sign_ok = np.sign(obs[ti]) == tsign
        dominant_ok = (dom_i == ti) and sign_ok
        lo, hi = boot_ci(scores[dom].to_numpy(), scores[ttrait].to_numpy(), n_boot, seed)

        pc_obs = profile_corr(obs, litvec)
        pc_lat = profile_corr(lat, litvec)

        print(f"\n[{dom}]  predicted dominant: {tsign:+d}{ttrait}   "
              f"(lit: {dict(zip(TRAITS, litvec))})")
        print("        " + "  ".join(f"{t}={o:+.2f}" for t, o in zip(TRAITS, obs))
              + "   <- observed (marker-based)")
        print("        " + "  ".join(f"{t}={o:+.2f}" for t, o in zip(TRAITS, lat))
              + "   <- latent-based")
        print(f"        dominant match: {'YES' if dominant_ok else 'no '} "
              f"(silicon argmax={'%+d%s' % (np.sign(obs[dom_i]), TRAITS[dom_i])}); "
              f"target r[{ttrait}]={obs[ti]:+.2f}  95% CI [{lo:+.2f},{hi:+.2f}]")
        print(f"        profile corr vs lit:  observed={pc_obs:+.2f}  latent={pc_lat:+.2f}")

    rmse = float(np.sqrt(np.nanmean((np.array(all_obs) - np.array(all_lit)) ** 2)))
    mean_pc = float(np.nanmean([profile_corr(o, l) for o, l in zip(all_obs, all_lit)]))
    print(f"\n  >>> matrix RMSE (observed vs lit) = {rmse:.3f}   "
          f"mean profile corr = {mean_pc:+.3f}")
    return {"rmse": rmse, "mean_profile_corr": mean_pc}


def degeneracy(df, meta):
    """Flag collapsed/flat per-scale response distributions (OOD failure mode)."""
    resp_col = next(c for c in ["samp_resp", "response", "pred_resp", "true_resp"]
                    if c in df.columns)
    m = df.merge(meta[["item_id", "scale"]], on="item_id", how="inner")
    print(f"\n{'-'*78}\nResponse-distribution sanity (per scale): mean / sd / %@extremes(1 or 5)")
    for scale, g in m.groupby("scale"):
        r = g[resp_col]
        ext = ((r == 1) | (r == SCALE_MAX)).mean()
        flag = "  <-- LOW VARIANCE" if r.std() < 0.4 else ""
        print(f"  {scale:5s}  mean={r.mean():.2f}  sd={r.std():.2f}  ext={ext:.0%}{flag}")


def main(args):
    meta = pd.read_csv(args.meta)
    lit = pd.read_csv(args.literature)
    sil = pd.read_csv(args.silicon)

    degeneracy(sil, meta)
    s_full = scale_scores(sil, meta)
    analyze(s_full, meta, lit, args.n_boot, args.seed, f"FULL  ({args.silicon})")

    if args.floor:
        flo = pd.read_csv(args.floor)
        s_flo = scale_scores(flo, meta)
        analyze(s_flo, meta, lit, args.n_boot, args.seed, f"FLOOR / baseline_a  ({args.floor})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--silicon", default="silicon_criterion_full.csv")
    p.add_argument("--floor", default="silicon_criterion_floor.csv")
    p.add_argument("--meta", default="data/criterion_meta.csv")
    p.add_argument("--literature", default="data/criterion_literature.csv")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
