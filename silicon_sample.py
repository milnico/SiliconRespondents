"""Generate a *silicon sample*: a synthetic respondent population.

Latents are drawn from the human factor-score distribution (MVN with the
empirical covariance of the CFA scores) so the inter-factor correlations Phi
can be reproduced. Each synthetic respondent answers every item by *sampling*
from the model's categorical (sampling supplies the item-unique variance a
deterministic map lacks). Items default to the held-out set, demonstrating that
the structure holds on items the model never trained on.
"""
import torch
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm

from train_simulator import build_prompt
from model.simulator import RespondentSimulator

TRAITS = ["O", "C", "E", "A", "N"]


def load_population_latents(latent_csv, n, seed, latents_file=None):
    # Reuse a fixed respondent population across runs (e.g. test-retest reliability:
    # same z, different response-sampling noise). If latents_file exists, load it;
    # otherwise draw fresh from the human factor-score MVN and persist it.
    if latents_file and os.path.exists(latents_file):
        z = np.load(latents_file)
        assert z.shape[0] >= n, f"{latents_file} has {z.shape[0]} < {n} latents"
        z = z[:n]
        print(f"Loaded {len(z)} fixed population latents from {latents_file}")
        return z.astype(np.float32)
    scores = pd.read_csv(latent_csv)[TRAITS].values
    mu = scores.mean(axis=0)
    cov = np.cov(scores, rowvar=False)
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(mu, cov, size=n)
    print(f"Population latent cov (human):\n{np.round(cov, 2)}")
    if latents_file:
        np.save(latents_file, z.astype(np.float32))
        print(f"Saved population latents -> {latents_file}")
    return z.astype(np.float32)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    # The latent draw is controlled by --seed; the per-item response sampling is
    # controlled by --sample_seed (defaults to --seed). Decoupling them lets us hold
    # the respondent population fixed while re-administering with fresh sampling noise.
    sample_seed = args.sample_seed if args.sample_seed is not None else args.seed
    rng = np.random.default_rng(sample_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    trait_text = args.mode == "baseline_b"

    # Items to administer (default: held-out items the model never trained on).
    items_df = pd.read_parquet(args.items_file)[["item_id", "item_text", "factor"]].drop_duplicates()
    items = list(items_df.itertuples(index=False))
    print(f"Administering {len(items)} items to {args.n} synthetic respondents.")

    z = load_population_latents(args.latent_csv, args.n, args.seed, args.latents_file)
    pids = [f"S{i:05d}" for i in range(args.n)]

    model = RespondentSimulator(model_name=args.model_name,
                                target_layers=args.target_layers, num_prefix=args.num_prefix)
    model.set_mode(args.mode)
    model.set_answer_tokens(tokenizer)
    model.prepare_trainable(device, dtype)
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint {args.checkpoint} ({len(sd)} tensors)")
    model.eval()

    z_t = torch.tensor(z, device=device, dtype=dtype)
    rows = []
    with torch.no_grad():
        for item in tqdm(items, desc="items"):
            # For prefix/no-trait modes the prompt is identical across respondents
            # (trait enters via the soft prefix); build it once and reuse.
            shared_prompt = None
            if not trait_text:
                enc = tokenizer(build_prompt(tokenizer, item.item_text, latents=None, trait_text=False),
                                return_tensors="pt").to(device)
                shared_prompt = (enc["input_ids"], enc["attention_mask"])

            for start in range(0, args.n, args.batch_size):
                zb = z_t[start:start + args.batch_size]
                bs = zb.size(0)
                if trait_text:
                    # baseline_b: persona is in the TEXT, so each respondent gets a
                    # distinct prompt encoding their own trait levels.
                    prompts = [build_prompt(tokenizer, item.item_text, latents=z[start + j].tolist(),
                                            trait_text=True, persona_style=args.persona_style)
                               for j in range(bs)]
                    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
                    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
                else:
                    input_ids = shared_prompt[0].expand(bs, -1)
                    attention_mask = shared_prompt[1].expand(bs, -1)
                logits = model(input_ids, attention_mask, zb) / args.temperature
                probs = torch.softmax(logits, -1).float().cpu().numpy()
                probs = probs / probs.sum(1, keepdims=True)
                u = rng.random((bs, 1))
                samp = (probs.cumsum(1) > u).argmax(1) + 1
                for j in range(bs):
                    gi = start + j
                    rows.append({
                        "participant_id": pids[gi],
                        "O": z[gi, 0], "C": z[gi, 1], "E": z[gi, 2], "A": z[gi, 3], "N": z[gi, 4],
                        "item_id": item.item_id, "item_text": item.item_text, "factor": item.factor,
                        "samp_resp": int(samp[j]),
                        # `true_resp`/`pred_resp` aliases so the eval scripts work unchanged.
                        "true_resp": int(samp[j]), "pred_resp": int(samp[j]),
                    })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    out.to_csv(args.output_file, index=False)
    print(f"Wrote silicon sample: {len(out)} rows -> {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/simulator_full.pt")
    parser.add_argument("--mode", type=str, default="full")
    parser.add_argument("--items_file", type=str, default="data/item_holdout.parquet")
    parser.add_argument("--latent_csv", type=str, default="latent_scores.csv")
    parser.add_argument("--output_file", type=str, default="silicon_sample.csv")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    # T~0.6 best matches human psychometrics (max factor congruence on the
    # plateau + closest Cronbach alpha, without the eigenvalue/alpha overshoot
    # that lower temperatures introduce). See the temperature sweep.
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--target_layers", type=int, default=12)
    parser.add_argument("--num_prefix", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for drawing the population latents z (the respondents)")
    parser.add_argument("--sample_seed", type=int, default=None,
                        help="seed for the per-item categorical response sampling only "
                             "(defaults to --seed). Vary this with --seed fixed for test-retest.")
    parser.add_argument("--latents_file", type=str, default=None,
                        help="path to a .npy of population latents; load if present else draw+save. "
                             "Lets multiple administrations share one fixed respondent population.")
    parser.add_argument("--persona_style", type=str, choices=["numeric", "verbal"], default="verbal",
                        help="baseline_b only: how the trait profile is written into the prompt")
    args = parser.parse_args()
    main(args)
