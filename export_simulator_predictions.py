import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import argparse
import os
from tqdm import tqdm
import numpy as np

from train_simulator import LatentSimulatorDataset
from model.simulator import RespondentSimulator


def export_predictions(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    print(f"Using device: {device}")
    rng = np.random.default_rng(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    trait_text = args.mode == "baseline_b"

    dataset = LatentSimulatorDataset(args.data_file, tokenizer, noise_std=0.0, trait_text=trait_text)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = RespondentSimulator(model_name=args.model_name,
                                target_layers=args.target_layers, num_prefix=args.num_prefix)
    model.set_mode(args.mode)
    model.set_answer_tokens(tokenizer)
    model.prepare_trainable(device, dtype)

    if args.checkpoint:
        print(f"Loading weights from {args.checkpoint}...")
        state_dict = torch.load(args.checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  loaded ({len(state_dict)} tensors); unexpected={len(unexpected)}")
    model.eval()

    all_probs = []
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            latents = batch["latent_traits"].to(device=device, dtype=dtype)

            logits = model(input_ids, attention_mask, latents) / args.temperature
            probs = torch.softmax(logits, dim=-1).to(torch.float32).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    all_probs = all_probs / all_probs.sum(axis=1, keepdims=True)

    df = dataset.data.copy()
    df["true_resp"] = df["response"]
    df["pred_resp"] = all_probs.argmax(axis=1) + 1  # mode (for accuracy reference)
    # Sampled response: this is what a silicon respondent "answers". Sampling
    # restores between-person variance that argmax destroys.
    cdf = all_probs.cumsum(axis=1)
    u = rng.random((len(df), 1))
    df["samp_resp"] = (cdf > u).argmax(axis=1) + 1
    for i in range(5):
        df[f"prob_{i + 1}"] = all_probs[:, i]

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output_file, index=False)
    print(f"Exported {len(df)} predictions to {args.output_file}")
    acc = (df["pred_resp"] == df["true_resp"]).mean()
    print(f"  argmax accuracy={acc:.3f}  "
          f"pred(samp) var={df.groupby('item_id')['samp_resp'].var().mean():.3f}  "
          f"true var={df.groupby('item_id')['true_resp'].var().mean():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--mode", type=str, default="full")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--target_layers", type=int, default=12)
    parser.add_argument("--num_prefix", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    export_predictions(args)
