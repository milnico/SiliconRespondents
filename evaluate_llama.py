import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr
import argparse
import os
from tqdm import tqdm

from model.latent_llama import LatentLlamaModel
from train_llama import LlamaPsychometricDataset

def evaluate_split(model, dataloader, device, split_name, target_tokens):
    model.eval()
    all_preds = []
    all_labels = []
    all_item_ids = []
    all_p_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            latent_traits = batch['latent_traits'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, latent_traits)
            last_token_logits = outputs.logits[:, -1, :]
            vocab_logits = last_token_logits[:, target_tokens]
            preds = torch.argmax(vocab_logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_item_ids.extend(batch['item_id'])
            all_p_ids.extend(batch['participant_id'])
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    df = pd.DataFrame({
        'participant_id': all_p_ids,
        'item_id': all_item_ids,
        'true_resp': all_labels,
        'pred_resp': all_preds
    })
    
    item_stats = df.groupby('item_id').agg({
        'true_resp': ['mean', 'var'],
        'pred_resp': ['mean', 'var']
    })
    
    mean_rmse = np.sqrt(np.mean((item_stats[('true_resp', 'mean')] - item_stats[('pred_resp', 'mean')])**2))
    var_rmse = np.sqrt(np.mean((item_stats[('true_resp', 'var')] - item_stats[('pred_resp', 'var')])**2))
    
    try:
        true_pivot = df.pivot(index='participant_id', columns='item_id', values='true_resp').fillna(2)
        pred_pivot = df.pivot(index='participant_id', columns='item_id', values='pred_resp').fillna(2)
        
        true_corr = true_pivot.corr().values
        pred_corr = pred_pivot.corr().values
        
        frob_norm = np.linalg.norm(true_corr - pred_corr)
    except Exception as e:
        frob_norm = float('nan')
        
    print(f"=== {split_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Item Mean RMSE: {mean_rmse:.4f}")
    print(f"Item Var RMSE:  {var_rmse:.4f}")
    print(f"Correlation Frobenius Norm: {frob_norm:.4f}")
    print("="*20)
    
    return {
        "acc": acc,
        "f1": f1,
        "mean_rmse": mean_rmse,
        "var_rmse": var_rmse,
        "frob_norm": frob_norm
    }

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = LatentLlamaModel(model_name=args.model_name, num_film_layers=6)
    
    save_path = os.path.join(args.output_dir, "llama_film_model.pt")
    if os.path.exists(save_path):
        state_dict = torch.load(save_path, map_location=device)
        filtered_dict = {k: v for k, v in state_dict.items() if "lora" in k or "latent_encoder" in k or "film_layers" in k}
        model.load_state_dict(filtered_dict, strict=False)
        print(f"Loaded {save_path} (filtered {len(filtered_dict)} trainable parameters)")
    else:
        print(f"Warning: {save_path} not found. Evaluating randomly initialized model/lora.")
        
    model.latent_encoder.to(device)
    model.film_layers.to(device)
    
    target_tokens = torch.tensor([16, 17, 18, 19, 20]).to(device)
    
    splits = {
        "Split A (Train)": "split_A_train.parquet",
        "Split B (New Participants)": "split_B_new_participants.parquet",
        "Split C (New Items)": "split_C_new_items.parquet",
        "Split D (New Participants & Items)": "split_D_new_both.parquet"
    }
    
    results = {}
    for name, file in splits.items():
        if not os.path.exists(file):
            print(f"{file} not found, skipping {name}")
            continue
            
        dataset = LlamaPsychometricDataset(file, tokenizer, noise_std=0.0)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        results[name] = evaluate_split(model, dataloader, device, name, target_tokens)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="unsloth/llama-3-8b-instruct-bnb-4bit")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
