import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import argparse
import os
from tqdm import tqdm

from model.latent_llama import LatentLlamaModel
from train_llama import LlamaPsychometricDataset

def export_split(model, dataloader, device, split_name, target_tokens):
    model.eval()
    all_preds = []
    all_labels = []
    all_item_ids = []
    all_p_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Exporting {split_name}"):
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
            
    df = pd.DataFrame({
        'participant_id': all_p_ids,
        'item_id': all_item_ids,
        'true_resp': all_labels,
        'pred_resp': all_preds
    })
    
    # +1 to make them 1-5 scale (assuming labels are 0-4 as per dataset setup)
    df['true_resp'] = df['true_resp'] + 1
    df['pred_resp'] = df['pred_resp'] + 1
    
    save_name = f"predictions_llama_{split_name.replace(' ', '_').split('_(')[0]}.csv"
    df.to_csv(save_name, index=False)
    print(f"Saved predictions to {save_name}")

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
        print(f"Warning: {save_path} not found. Exporting with randomly initialized model.")
        
    model.latent_encoder.to(device)
    model.film_layers.to(device)
    
    target_tokens = torch.tensor([16, 17, 18, 19, 20]).to(device)
    
    # We explicitly EXCLUDE Split A to save hours of inference time
    splits = {
        "Split C Fast": "split_C_new_items.parquet",
    }
    
    for name, file in splits.items():
        print(f"\nEvaluating {name}...")
        df = pd.read_parquet(file)
        df = df.sample(n=5000, random_state=42) # Fast subset!
        
        save_name = f"predictions_llama_{name.replace(' ', '_').split('_(')[0]}.csv"
        if not os.path.exists(file):
            print(f"{file} not found, skipping {name}")
            continue
            
        dataset = LlamaPsychometricDataset(df, tokenizer, noise_std=0.0)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        export_split(model, dataloader, device, name, target_tokens)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="unsloth/llama-3-8b-instruct-bnb-4bit")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
