"""Self-judge scoring for the trait-conditioned text-generation demo (§7.11).

For each generation produced by `latent_intervention.py --task text --all_traits`,
we ask the SAME backbone model in `baseline_a` mode (no trait conditioning —
neutral reader) to rate the described person on each Big Five trait via a
standard Likert agreement probe. The digit logits over {1..5} give a soft
"perceived trait" score per probe.

This isolates the *steering-fidelity* question: when we condition the
generator on a trait vector z, does the perceived trait in the resulting text
actually move on-target, and stay roughly stable off-target?

Outputs:
  results/text_steering_scored.csv   per-generation 5-vector of perceived-trait E[r].
  results/text_steering_summary.md   on-target effect-sizes + (swept × measured) leakage matrix.

Usage:
  python 09_text_scoring.py \
      --gens results/text_steering_generations.csv \
      --checkpoint checkpoints/qwen/seed0/simulator_full.pt
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from model.simulator import RespondentSimulator

TRAITS = ["O", "C", "E", "A", "N"]
TRAIT_NAMES_LONG = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

# Standard third-person Likert probes. The model reads a generation and rates
# agreement with each statement on 1..5. Higher = perceived more of that trait.
PROBES = {
    "O": "This person seems open to new experiences and ideas.",
    "C": "This person seems conscientious and well-organized.",
    "E": "This person seems extraverted and outgoing.",
    "A": "This person seems agreeable, warm, and considerate.",
    "N": "This person seems anxious or emotionally unstable.",
}


def build_judge_prompt(tokenizer, generation_text, probe_statement):
    """Probe prompt for the neutral self-judge."""
    user = (
        f"Read this description of a person:\n\n\"{generation_text}\"\n\n"
        "On a scale of 1 to 5 (1 = Strongly Disagree, 5 = Strongly Agree), "
        f"how much do you agree:\n\"{probe_statement}\"\n"
        "Answer with a single number."
    )
    messages = [
        {"role": "system", "content": "You are reading a person's description and rating their personality."},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def score_generations(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = RespondentSimulator(model_name=args.model_name,
                                target_layers=args.target_layers, num_prefix=args.num_prefix)
    model.set_mode("baseline_a")   # neutral reader: no trait conditioning, no LoRA
    model.set_answer_tokens(tokenizer)
    model.prepare_trainable(device, dtype)
    if args.checkpoint:
        # Loads trainable weights, but baseline_a bypasses prefix/LoRA so the
        # backbone is effectively the frozen base model — a clean judge.
        sd = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(sd, strict=False)
    model.eval()

    gens = pd.read_csv(args.gens)
    print(f"Scoring {len(gens)} generations against {len(PROBES)} probes "
          f"= {len(gens) * len(PROBES)} forward passes...")

    scored_rows = []
    levels_1to5 = np.arange(1, 6, dtype=np.float32)
    with torch.no_grad():
        for i, r in gens.iterrows():
            text = str(r["text"]).strip()
            if not text:
                continue
            row_out = {k: r[k] for k in ["trait_idx", "trait", "level", "prompt_id", "prompt", "sample", "n_tokens"]}
            row_out["text"] = text
            for tname, probe_stmt in PROBES.items():
                prompt = build_judge_prompt(tokenizer, text, probe_stmt)
                enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                # zb is ignored in baseline_a mode, but the forward signature wants a 5-vector.
                zb = torch.zeros(1, 5, dtype=dtype, device=device)
                logits = model(enc["input_ids"], enc["attention_mask"], zb)
                p = torch.softmax(logits, -1).squeeze(0).cpu().float().numpy()
                exp = float((p * levels_1to5).sum())
                row_out[f"perceived_{tname}"] = exp
            scored_rows.append(row_out)
            if (i + 1) % 25 == 0 or (i + 1) == len(gens):
                print(f"  [{i + 1}/{len(gens)}] done")
    df = pd.DataFrame(scored_rows)
    df.to_csv(args.scored_out, index=False)
    print(f"Wrote {args.scored_out}")
    return df


_REFUSAL_RE = __import__("re").compile(r"\b(AI|language model|assistant)\b",
                                       __import__("re").IGNORECASE)


def _is_refusal(text):
    return bool(_REFUSAL_RE.search(str(text)))


def _delta_matrix(df, full_names):
    """(swept × measured) effect-size matrix at z=+2 minus z=−2."""
    deltas = np.full((5, 5), np.nan)
    n_per_cell = np.full((5, 2), 0, dtype=int)
    for i, swept_full in enumerate(full_names):
        sub_h = df[(df.trait == swept_full) & (df.level == 2.0)]
        sub_l = df[(df.trait == swept_full) & (df.level == -2.0)]
        n_per_cell[i] = [len(sub_l), len(sub_h)]
        for j, meas in enumerate(TRAITS):
            mh = sub_h[f"perceived_{meas}"].mean()
            ml = sub_l[f"perceived_{meas}"].mean()
            deltas[i, j] = float(mh - ml)
    return deltas, n_per_cell


def _emit_delta_table(lines, deltas, full_names, label):
    lines.append(f"## {label}\n")
    head = "| swept ↓ / measured → | " + " | ".join(TRAITS) + " | (on-target Δ) |"
    sep  = "|" + "|".join(["---"] * (2 + len(TRAITS))) + "|"
    lines.append(head); lines.append(sep)
    for i, swept_full in enumerate(full_names):
        cells_row = []
        for j in range(5):
            d = deltas[i, j]
            star = " ✓" if i == j and not np.isnan(d) and np.abs(d) == np.nanmax(np.abs(deltas[i])) else ""
            cells_row.append(f"{d:+.2f}{star}" if not np.isnan(d) else " n/a")
        lines.append(f"| **{TRAITS[i]} ({swept_full})** | " + " | ".join(cells_row)
                     + f" | **{deltas[i, i]:+.2f}** |")
    diag = np.diag(deltas)
    diag_wins = int(np.sum(np.abs(diag) == np.nanmax(np.abs(deltas), axis=1)))
    on = float(np.mean(np.abs(diag)))
    off = float(np.mean(np.abs(deltas - np.diag(diag))))
    lines.append(f"\nDiagonal dominance (on-target |Δ| is the row-max): **{diag_wins}/5**. "
                 f"Mean |on-target Δ|: {on:.2f}; mean |off-target Δ|: {off:.2f}.\n")


def aggregate_and_summarize(df, out_md):
    """Effect-size table + (swept × measured) leakage matrix. Reports both the
    full 225 generations and a refusal-filtered subset to make the chat-template
    confound transparent."""
    full_names = TRAIT_NAMES_LONG  # ["Openness", "Conscientiousness", ...]
    df = df.copy()
    df["is_refusal"] = df["text"].apply(_is_refusal)
    n_refuse = int(df["is_refusal"].sum())

    lines = ["# Trait-conditioned text-steering — self-judge scoring (§7.11)\n"]
    lines.append("Each cell shows the **change in perceived trait** when steering the generator from "
                 "z = −2 to z = +2 on the *swept trait* (rows), as judged by the same backbone in "
                 "baseline_a mode (no trait conditioning — neutral reader). Diagonal entries are the "
                 "on-target effect; off-diagonal entries measure leakage.\n")
    lines.append(f"**Chat-template / RLHF refusal rate**: {n_refuse}/{len(df)} "
                 f"({100*n_refuse/len(df):.0f}%) of generations contain 'AI / language model / "
                 f"assistant'. The training loss never penalised this prior (only digit-token logits "
                 f"were supervised), so the assistant persona leaks into open-ended free text in many "
                 f"cells. We report both the full subset and the refusal-filtered subset below.\n")

    # Full 225 generations
    deltas_all, n_all = _delta_matrix(df, full_names)
    _emit_delta_table(lines, deltas_all, full_names,
                      f"Full set ({len(df)} generations) — Δ self-judge units")

    # Refusal-filtered subset
    clean = df[~df["is_refusal"]].copy()
    deltas_c, n_c = _delta_matrix(clean, full_names)
    _emit_delta_table(lines, deltas_c, full_names,
                      f"Refusal-filtered subset ({len(clean)} generations) — Δ self-judge units")

    # Cohen's-d-style on-target effect (on refusal-filtered subset, the more interpretable one)
    lines.append("## On-target effect size (Cohen's-d-style, refusal-filtered subset)\n")
    lines.append("| swept | n(−2) | n(+2) | mean @ z=−2 | mean @ z=+2 | Δ | pooled SD | d |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, swept_full in enumerate(full_names):
        col = f"perceived_{TRAITS[i]}"
        h = clean[(clean.trait == swept_full) & (clean.level == 2.0)][col].to_numpy()
        l = clean[(clean.trait == swept_full) & (clean.level == -2.0)][col].to_numpy()
        if len(h) < 2 or len(l) < 2:
            lines.append(f"| {TRAITS[i]} | {len(l)} | {len(h)} | — | — | — | — | — |")
            continue
        delta = float(h.mean() - l.mean())
        pooled_sd = float(np.sqrt(((h.var(ddof=1) * (len(h) - 1)) + (l.var(ddof=1) * (len(l) - 1)))
                                  / (len(h) + len(l) - 2))) if (len(h) + len(l) - 2) > 0 else float("nan")
        d = float(delta / pooled_sd) if pooled_sd > 0 else float("nan")
        lines.append(f"| {TRAITS[i]} | {len(l)} | {len(h)} | {l.mean():.2f} | {h.mean():.2f} | "
                     f"{delta:+.2f} | {pooled_sd:.2f} | **{d:+.2f}** |")

    # Curated qualitative table: 1 generation per (trait × level), picked from
    # the refusal-filtered subset if available, else the raw set.
    lines.append("\n## Curated qualitative examples (first non-refusal sample per cell)\n")
    lines.append("| trait | z=−2 | z=0 | z=+2 |")
    lines.append("|---|---|---|---|")
    for i, swept_full in enumerate(full_names):
        cells_row = [f"**{TRAITS[i]} ({swept_full})**"]
        for level in [-2.0, 0.0, 2.0]:
            pool = clean[(clean.trait == swept_full) & (clean.level == level)]
            if len(pool) == 0:
                pool = df[(df.trait == swept_full) & (df.level == level)]
            text = (pool.iloc[0]["text"] if len(pool) else "—")
            text = str(text).replace("|", "/").replace("\n", " ")
            if len(text) > 160:
                text = text[:157] + "..."
            cells_row.append(text)
        lines.append("| " + " | ".join(cells_row) + " |")

    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")


def main(args):
    df = score_generations(args)
    aggregate_and_summarize(df, args.summary_out)
    # Echo summary to stdout
    print("\n" + open(args.summary_out).read())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gens", default="results/text_steering_generations.csv")
    p.add_argument("--scored_out", default="results/text_steering_scored_llama.csv")
    p.add_argument("--summary_out", default="results/text_steering_summary_llama.md")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--checkpoint", default="checkpoints/qwen/seed0/simulator_full.pt")
    p.add_argument("--target_layers", type=int, default=12)
    p.add_argument("--num_prefix", type=int, default=8)
    args = p.parse_args()
    main(args)
