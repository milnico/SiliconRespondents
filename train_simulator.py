import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import pandas as pd
import numpy as np
import argparse
import os
import random
from tqdm import tqdm

from model.simulator import RespondentSimulator

TRAIT_NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def _verbal_level(z):
    """Map a standardized trait value to a natural-language level."""
    if z >= 1.0:
        return "very high"
    if z >= 0.5:
        return "high"
    if z > -0.5:
        return "average"
    if z > -1.0:
        return "low"
    return "very low"


def build_prompt(tokenizer, item_text, latents=None, trait_text=False, persona_style="numeric"):
    """Build the chat prompt whose next generated token is the Likert digit.

    `trait_text=True` writes the trait profile into the prompt (baseline_b). With
    `persona_style="verbal"` the profile is natural language (high/low levels),
    which frozen instruct models read far better than raw z-scores. For the
    prefix/LoRA modes the latent is injected as a soft prefix instead, so the
    text profile is omitted.
    """
    profile = ""
    if trait_text and latents is not None:
        if persona_style == "verbal":
            lines = "\n".join(f"- {n}: {_verbal_level(v)}" for n, v in zip(TRAIT_NAMES, latents))
            profile = ("You are role-playing a specific person with this personality profile "
                       f"(Big Five):\n{lines}\n\nAnswer the question as this person would.\n\n")
        else:
            lines = "\n".join(f"{n}: {v:.2f}" for n, v in zip(TRAIT_NAMES, latents))
            profile = f"Participant Profile\n{lines}\n\n"

    user = (
        f"{profile}Question:\n{item_text}\n\n"
        "On a scale of 1 to 5 (1 = Strongly Disagree, 5 = Strongly Agree), "
        "how much do you agree?\nAnswer with a single number."
    )
    messages = [
        {"role": "system", "content": "You are a psychometric respondent simulator."},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class LatentSimulatorDataset(Dataset):
    def __init__(self, parquet_file, tokenizer, max_length=128, noise_std=0.1, trait_text=False):
        self.data = pd.read_parquet(parquet_file)
        # Drop missing responses (0) to keep CrossEntropy in range.
        self.data = self.data[self.data["response"] > 0].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.noise_std = noise_std
        self.trait_text = trait_text

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = int(row["response"]) - 1  # 0..4

        latents = torch.tensor([row["O"], row["C"], row["E"], row["A"], row["N"]], dtype=torch.float32)
        if self.noise_std > 0:
            latents = latents + torch.randn_like(latents) * self.noise_std

        prompt = build_prompt(self.tokenizer, row["item_text"],
                              latents=latents.tolist(), trait_text=self.trait_text)
        encoding = self.tokenizer(
            prompt, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "latent_traits": latents,
            "labels": torch.tensor(label, dtype=torch.long),
        }


@torch.no_grad()
def evaluate(model, loader, device, dtype, criterion, max_batches=None):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        latents = batch["latent_traits"].to(device=device, dtype=dtype)
        labels = batch["labels"].to(device)
        logits = model(input_ids, attention_mask, latents)
        total_loss += criterion(logits, labels).item()
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.size(0)
    model.train()
    n = max(1, (i + 1) if max_batches is None else min(max_batches, i + 1))
    return total_loss / n, correct / max(1, total)


def _seed_worker(worker_id):
    # Each DataLoader worker derives its numpy/random seed from torch's
    # per-worker base seed (itself a function of the global manual_seed).
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def train(args):
    # Seed every RNG BEFORE building the model/dataset so weight init, data
    # shuffling and latent noise are controlled by --seed. Different seeds give
    # distinct-but-controlled runs (for between-seed variance); we deliberately
    # do NOT force deterministic cudnn algorithms (can error / slow training).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | seed: {args.seed}")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    trait_text = args.mode == "baseline_b"

    train_dataset = LatentSimulatorDataset(args.train_data, tokenizer,
                                           noise_std=args.noise_std, trait_text=trait_text)
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=True,
                              generator=loader_gen, worker_init_fn=_seed_worker)

    val_loader = None
    if args.val_data and os.path.exists(args.val_data):
        val_dataset = LatentSimulatorDataset(args.val_data, tokenizer,
                                             noise_std=0.0, trait_text=trait_text)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=4)

    model = RespondentSimulator(model_name=args.model_name,
                                target_layers=args.target_layers, num_prefix=args.num_prefix)
    model.set_mode(args.mode)
    model.set_answer_tokens(tokenizer)
    model.prepare_trainable(device, dtype)

    criterion = nn.CrossEntropyLoss()
    params = model.trainable_parameters()
    n_train = sum(p.numel() for p in params)
    print(f"Trainable parameters: {n_train:,}")

    if args.mode == "baseline_a" or args.mode == "baseline_b":
        # Nothing to train in these prompt-only baselines except... nothing.
        # They are evaluated zero-shot through the frozen LM head.
        print(f"Mode {args.mode} has no trainable conditioning modules; "
              "evaluating zero-shot (skipping optimization).")
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f"simulator_{args.mode}.pt")
        torch.save({}, save_path)
        return

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    # Scheduler horizon is the FULL run (args.epochs) so the cosine LR is correct
    # across resumes, even when a single invocation runs fewer epochs.
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.03 * total_steps), num_training_steps=total_steps)

    best_val = float("inf")
    start_epoch = 0
    global_step = 0
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"simulator_{args.mode}.pt")
    state_path = os.path.join(args.output_dir, f"simulator_{args.mode}_state.pt")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["next_epoch"]
        global_step = ckpt["global_step"]
        best_val = ckpt.get("best_val", float("inf"))
        print(f"Resumed from {args.resume}: starting at epoch {start_epoch + 1}, "
              f"global_step={global_step}, best_val={best_val:.4f}")

    # How many epochs to run in THIS invocation.
    end_epoch = args.epochs if args.run_epochs <= 0 else min(args.epochs, start_epoch + args.run_epochs)

    model.train()
    for epoch in range(start_epoch, end_epoch):
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            latents = batch["latent_traits"].to(device=device, dtype=dtype)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, latents)
            loss = criterion(logits, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running += loss.item()
            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                print(f"Hit max_steps={args.max_steps}; stopping early.")
                break
            if global_step % args.log_every == 0:
                pbar.set_postfix({"loss": running / args.log_every,
                                  "gnorm": float(grad_norm), "lr": scheduler.get_last_lr()[0]})
                running = 0.0

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, device, dtype, criterion,
                                         max_batches=args.val_batches)
            print(f"[epoch {epoch + 1}] item-holdout loss={val_loss:.4f} acc={val_acc:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                _save(model, save_path)
                print(f"  new best -> saved to {save_path}")
        else:
            _save(model, save_path)

        # Always persist full training state so the run can be resumed.
        _save_state(model, optimizer, scheduler, save_state_path=state_path,
                    next_epoch=epoch + 1, global_step=global_step, best_val=best_val)

    print(f"Ran epochs {start_epoch + 1}..{end_epoch} of {args.epochs}. "
          f"Best (inference) checkpoint: {save_path}; resume state: {state_path}")


def _trainable_dict(model):
    sd = model.state_dict()
    return {k: v for k, v in sd.items()
            if ("lora_A" in k) or ("lora_B" in k) or ("gate_mlp" in k)
            or k.startswith("trait_encoder.") or k.startswith("prefix_proj.")}


def _save(model, save_path):
    torch.save(_trainable_dict(model), save_path)


def _save_state(model, optimizer, scheduler, save_state_path, next_epoch, global_step, best_val):
    torch.save({
        "model": _trainable_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "next_epoch": next_epoch,
        "global_step": global_step,
        "best_val": best_val,
    }, save_state_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="data/train.parquet")
    parser.add_argument("--val_data", type=str, default="data/item_holdout.parquet")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--mode", type=str,
                        choices=["baseline_a", "baseline_b", "baseline_c", "full"], default="full")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3, help="total cosine-schedule horizon")
    parser.add_argument("--run_epochs", type=int, default=0,
                        help="epochs to run THIS invocation (0 = run to horizon)")
    parser.add_argument("--resume", type=str, default="",
                        help="path to a *_state.pt to resume optimizer/scheduler/epoch")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--target_layers", type=int, default=12)
    parser.add_argument("--num_prefix", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=0, help="0 = full epochs")
    parser.add_argument("--val_batches", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds init/shuffle/noise for reproducible, distinct runs")
    args = parser.parse_args()
    train(args)
