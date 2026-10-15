import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import argparse
import os

from model.latent_sentence_transformer import LatentConditionedModel
from model.baselines import BaselineMLP
from train import PsychometricDataset

def export_split(model, dataloader, device, output_file):
    model.eval()
    all_preds = []
    all_labels = []
    all_item_ids = []
    all_p_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            latent_traits = batch['latent_traits'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask, latent_traits)
            preds = torch.argmax(logits, dim=1)
            
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
    
    df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    splits = {
        "Split_A": "split_A_train.parquet",
        "Split_B": "split_B_new_participants.parquet",
        "Split_C": "split_C_new_items.parquet"
    }
    
    for model_type in ["film", "mlp"]:
        if model_type == "film":
            model = LatentConditionedModel(model_name=model_name)
        else:
            model = BaselineMLP(model_name=model_name)
            
        save_path = os.path.join("checkpoints", f"{model_type}_model.pt")
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.to(device)
        
        for name, file in splits.items():
            dataset = PsychometricDataset(file, tokenizer, noise_std=0.0)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
            output_file = f"predictions_{model_type}_{name}.csv"
            export_split(model, dataloader, device, output_file)

if __name__ == "__main__":
    main()
