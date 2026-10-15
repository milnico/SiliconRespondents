import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os

from model.latent_sentence_transformer import LatentConditionedModel
from model.baselines import BaselineMLP

class PsychometricDataset(Dataset):
    def __init__(self, parquet_file, tokenizer, max_length=64, noise_std=0.0):
        self.data = pd.read_parquet(parquet_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.noise_std = noise_std
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['item_text']
        # Responses are 1-5, convert to 0-4 for cross entropy
        label = int(row['response']) - 1
        
        latents = torch.tensor([row['O'], row['C'], row['E'], row['A'], row['N']], dtype=torch.float32)
        if self.noise_std > 0:
            latents += torch.randn_like(latents) * self.noise_std
            
        encoding = self.tokenizer(
            text,
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
    
    train_dataset = PsychometricDataset(args.train_data, tokenizer, noise_std=args.noise_std)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    if args.model_type == "film":
        model = LatentConditionedModel(model_name=args.model_name)
    else:
        model = BaselineMLP(model_name=args.model_name)
        
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
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
            logits = model(input_ids, attention_mask, latent_traits)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1), 'acc': correct / total})
            
        print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss/len(train_loader):.4f}, Train Acc: {correct/total:.4f}")
        
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{args.model_type}_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="split_A_train.parquet")
    parser.add_argument("--model_type", type=str, choices=["film", "mlp"], default="film")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    train(args)
