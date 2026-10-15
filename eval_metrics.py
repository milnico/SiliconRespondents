import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, f1_score, log_loss
import json

def compute_metrics(true_labels, pred_probs):
    """
    true_labels: array of shape (N,)
    pred_probs: array of shape (N, 5)
    """
    preds = np.argmax(pred_probs, axis=1)
    
    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='macro')
    ce = log_loss(true_labels, pred_probs)
    
    return {"accuracy": acc, "macro_f1": f1, "cross_entropy": ce}

def compute_item_stats(df, true_col, pred_col):
    item_stats = []
    items = df['item_id'].unique()
    
    for item in items:
        item_df = df[df['item_id'] == item]
        true_mean = item_df[true_col].mean()
        pred_mean = item_df[pred_col].mean()
        
        true_var = item_df[true_col].var()
        pred_var = item_df[pred_col].var()
        
        item_stats.append({
            'item_id': item,
            'true_mean': true_mean,
            'pred_mean': pred_mean,
            'true_var': true_var,
            'pred_var': pred_var
        })
        
    stats_df = pd.DataFrame(item_stats)
    rmse_mean = np.sqrt(((stats_df['true_mean'] - stats_df['pred_mean'])**2).mean())
    rmse_var = np.sqrt(((stats_df['true_var'] - stats_df['pred_var'])**2).mean())
    
    return {"rmse_item_means": rmse_mean, "rmse_item_variances": rmse_var}

def main(args):
    print(f"Loading predictions from {args.preds_file}...")
    # This assumes we have a CSV with true_labels, pred_probs, participant_id, item_id
    df = pd.read_csv(args.preds_file)
    
    # Example logic:
    # df['true_resp'] is the actual 1-5
    # df['pred_resp'] is the argmax
    # df['prob_1'] to df['prob_5'] are the probabilities
    
    prob_cols = ['prob_1', 'prob_2', 'prob_3', 'prob_4', 'prob_5']
    if all(c in df.columns for c in prob_cols):
        true_labels = df['true_resp'].values - 1
        pred_probs = df[prob_cols].values
        
        base_metrics = compute_metrics(true_labels, pred_probs)
        print("Base Metrics:", json.dumps(base_metrics, indent=2))
        
        item_metrics = compute_item_stats(df, 'true_resp', 'pred_resp')
        print("Item Stats:", json.dumps(item_metrics, indent=2))
        
    else:
        print("Predictions file is missing probability columns. Please ensure it outputs prob_1 to prob_5.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
