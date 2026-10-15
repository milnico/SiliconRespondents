"""Latent intervention: sweep one Big Five trait and observe the effect.

Two modes:
  * likert  -- response distribution over 1..5 for a questionnaire item
  * text    -- free-text generation in the *voice* of the trait profile
              (with --all_traits, structured sweep across all 5 traits × 3 levels
              × N prompts × N samples, written to a CSV for downstream scoring)
"""
import argparse
import csv
import os
import torch
from transformers import AutoTokenizer

from model.simulator import RespondentSimulator
from train_simulator import build_prompt, TRAIT_NAMES

SWEEP = [-2.0, -1.0, 0.0, 1.0, 2.0]
TEXT_SWEEP = [-2.0, 0.0, 2.0]
TEXT_PROMPTS = [
    # Note: an earlier prompt ("Tell me about your weekend plans.") was rejected
    # by Qwen's RLHF prior 100% of the time ("As an AI, I don't have weekends...").
    # Replaced with a roleplay-friendlier framing.
    "Describe a typical Saturday for you, from morning to evening.",
    "Describe how you'd handle starting a new job.",
    "What do you do when you're at a party where you don't know anyone?",
]


def load_model(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = RespondentSimulator(model_name=args.model_name,
                                target_layers=args.target_layers, num_prefix=args.num_prefix)
    model.set_mode(args.mode)
    model.set_answer_tokens(tokenizer)
    model.prepare_trainable(device, torch.bfloat16)
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(f"Loaded {args.checkpoint} ({len(sd)} tensors)")
    model.eval()
    return model, tokenizer


def trait_vector(trait_idx, val, device):
    z = torch.zeros(1, 5, dtype=torch.bfloat16, device=device)
    z[0, trait_idx] = val
    return z


def run_likert(model, tokenizer, args, device):
    print(f"\nLikert sweep of '{TRAIT_NAMES[args.trait_idx]}' on: '{args.item_text}'")
    print(f"{'value':>6} | " + "  ".join(f"P{i}" for i in range(1, 6)) + "   E[resp]")
    prompt = build_prompt(tokenizer, args.item_text, latents=None,
                          trait_text=(args.mode == "baseline_b"))
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        for val in SWEEP:
            z = trait_vector(args.trait_idx, val, device)
            logits = model(enc["input_ids"], enc["attention_mask"], z)
            p = torch.softmax(logits, -1).squeeze(0).cpu().float().numpy()
            exp = float((p * [1, 2, 3, 4, 5]).sum())
            print(f"{val:>6.1f} | " + "  ".join(f"{x:.2f}" for x in p) + f"   {exp:.2f}")


SWEEP_SYS = (
    "You are a real human person being interviewed about your life. "
    "Answer each question in the first person, in 1-3 sentences, sharing what you "
    "actually do, feel, or think. NEVER say you are an AI, language model, or "
    "assistant — you are a human person. Do not break character."
)


def _gen_one(model, tokenizer, prompt, z, max_new_tokens, device, seed,
             system_prompt="You are simulating a person. Write in the first person."):
    """Generate one completion under trait vector z. Returns (text, n_tokens)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(chat, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    out = model.generate_text(
        enc["input_ids"], enc["attention_mask"], z,
        max_new_tokens=max_new_tokens, do_sample=True,
        temperature=0.8, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    n_tok = int(out.shape[1])
    return text, n_tok


def run_text(model, tokenizer, args, device):
    """Single-trait, two-level demo (the original smoke test)."""
    trait = TRAIT_NAMES[args.trait_idx]
    print(f"\nText generation sweeping '{trait}'. Prompt: '{args.text_prompt}'")
    for val in [-2.0, 2.0]:
        z = trait_vector(args.trait_idx, val, device)
        text, _ = _gen_one(model, tokenizer, args.text_prompt, z,
                           args.max_new_tokens, device, seed=0)
        tag = "LOW" if val < 0 else "HIGH"
        print(f"\n--- {trait} = {val:+.1f} ({tag}) ---\n{text}")


def run_text_sweep(model, tokenizer, args, device):
    """Structured sweep: 5 traits × 3 levels × N prompts × N samples → CSV.

    Used as the data source for the §7.11 free-text steering demo. Generations
    are scored downstream by `09_text_scoring.py` via the self-judge.
    """
    os.makedirs(os.path.dirname(args.text_out) or ".", exist_ok=True)
    rows = []
    total = 5 * len(TEXT_SWEEP) * len(TEXT_PROMPTS) * args.n_samples
    done = 0
    for trait_idx, trait in enumerate(TRAIT_NAMES):
        for level in TEXT_SWEEP:
            z = trait_vector(trait_idx, level, device)
            for pid, prompt in enumerate(TEXT_PROMPTS):
                for s in range(args.n_samples):
                    # Deterministic per-cell seed so the CSV is reproducible but each cell still varies.
                    seed = 10_000 * trait_idx + 1_000 * int(level + 5) + 100 * pid + s
                    text, n_tok = _gen_one(model, tokenizer, prompt, z,
                                           args.max_new_tokens, device, seed=seed,
                                           system_prompt=SWEEP_SYS)
                    rows.append({
                        "trait_idx": trait_idx, "trait": trait, "level": level,
                        "prompt_id": pid, "prompt": prompt, "sample": s,
                        "n_tokens": n_tok, "text": text,
                    })
                    done += 1
                    if done % 10 == 0 or done == total:
                        print(f"  [{done:>3}/{total}] {trait}={level:+.0f} pid={pid} s={s}")
    with open(args.text_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} generations -> {args.text_out}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args, device)
    if args.task in ("likert", "both"):
        run_likert(model, tokenizer, args, device)
    if args.task in ("text", "both"):
        if args.all_traits:
            run_text_sweep(model, tokenizer, args, device)
        else:
            run_text(model, tokenizer, args, device)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--checkpoint", type=str, default="checkpoints/simulator_full.pt")
    p.add_argument("--mode", type=str, default="full")
    p.add_argument("--task", type=str, choices=["likert", "text", "both"], default="likert")
    p.add_argument("--item_text", type=str, default="I am the life of the party.")
    p.add_argument("--text_prompt", type=str,
                   default="Describe how you feel about going to a big party this weekend.")
    p.add_argument("--trait_idx", type=int, default=2, help="0=O,1=C,2=E,3=A,4=N")
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--target_layers", type=int, default=12)
    p.add_argument("--num_prefix", type=int, default=8)
    p.add_argument("--all_traits", action="store_true",
                   help="text mode: structured sweep over all 5 traits, 3 levels, 3 prompts, N samples")
    p.add_argument("--n_samples", type=int, default=5,
                   help="text mode (--all_traits): generations per (trait, level, prompt) cell")
    p.add_argument("--text_out", type=str, default="results/text_steering_generations.csv",
                   help="output CSV for the structured text sweep")
    args = p.parse_args()
    main(args)
