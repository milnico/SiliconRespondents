import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os

from model.latent_llama import LatentLlamaModel

class LlamaPsychometricDataset(Dataset):
    def __init__(self, data_df, tokenizer, max_length=128, noise_std=0.0):
        self.data = data_df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.noise_std = noise_std
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['item_text']
        label = int(row['response']) - 1
        
        latents = torch.tensor([row['O'], row['C'], row['E'], row['A'], row['N']], dtype=torch.bfloat16)
        if self.noise_std > 0:
            latents += torch.randn_like(latents) * self.noise_std
            
        messages = [
            {"role": "system", "content": "You are a helpful assistant predicting responses to psychometric tests."},
            {"role": "user", "content": f"Below is a psychometric questionnaire item. Respond with a single digit between 1 and 5.\n\nItem: {text}\nResponse:"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        encoding = self.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'latent_traits': latents,
            'labels': torch.tensor(label, dtype=torch.long),
            'item_id': row['item_id'],
            'participant_id': row['participant_id']
        }

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Fast Sampling!
    print("Loading full dataset and sampling 5%...")
    full_df = pd.read_parquet(args.train_data)
    fast_df = full_df.sample(frac=0.05, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(fast_df)} items for fast training.")
    
    train_dataset = LlamaPsychometricDataset(fast_df, tokenizer, noise_std=args.noise_std)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    model = LatentLlamaModel(model_name=args.model_name, num_film_layers=6)
    model.latent_encoder.to(device)
    model.film_layers.to(device)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    target_tokens = torch.tensor([16, 17, 18, 19, 20]).to(device)
    
    # Overwrite old loss log
    with open("fast_loss_log.csv", "w") as f:
        f.write("epoch,step,loss,acc\n")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            latent_traits = batch['latent_traits'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask, latent_traits)
            last_token_logits = outputs.logits[:, -1, :]
            vocab_logits = last_token_logits[:, target_tokens]
            
            loss = criterion(vocab_logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(vocab_logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if pbar.n % 10 == 0:
                with open("fast_loss_log.csv", "a") as f:
                    f.write(f"{epoch+1},{pbar.n},{total_loss / (pbar.n + 1):.4f},{correct / total:.4f}\n")
            
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1), 'acc': correct / total})
            
        print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss/len(train_loader):.4f}, Train Acc: {correct/total:.4f}")
        
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "llama_film_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="split_A_train.parquet")
    parser.add_argument("--model_name", type=str, default="unsloth/llama-3-8b-instruct-bnb-4bit")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    train(args)
