import pandas as pd
import numpy as np
import os

def main():
    print("Loading responses_long.parquet...")
    df = pd.read_parquet("responses_long.parquet")
    
    # Get unique items and participants
    all_items = df['item_id'].unique()
    all_participants = df['participant_id'].unique()
    
    # 1. Select holdout items (4 per factor -> 20 total). More items per factor
    #    gives a credible 5-factor structure test on the held-out ("new") items.
    item_factors = df[['item_id', 'factor']].drop_duplicates()
    holdout_items = []

    # Fixed random seed for reproducible splits (set once before the loop).
    np.random.seed(42)
    for factor in ['O', 'C', 'E', 'A', 'N']:
        factor_items = item_factors[item_factors['factor'] == factor]['item_id'].tolist()
        holdout_items.extend(np.random.choice(factor_items, 4, replace=False))
        
    train_items = [i for i in all_items if i not in holdout_items]
    
    print(f"Total items: {len(all_items)}")
    print(f"Train items: {len(train_items)}")
    print(f"Holdout items: {len(holdout_items)}")
    
    # 2. Split participants
    np.random.seed(42)
    shuffled_p = np.random.permutation(all_participants)
    split_idx = int(len(shuffled_p) * 0.8)
    
    train_p = shuffled_p[:split_idx]
    holdout_p = shuffled_p[split_idx:]
    
    print(f"Total participants: {len(all_participants)}")
    print(f"Train participants: {len(train_p)}")
    print(f"Holdout participants: {len(holdout_p)}")
    
    # 3. Create DataFrames
    # Train: 80% participants, 40 items
    train_df = df[(df['participant_id'].isin(train_p)) & (df['item_id'].isin(train_items))]
    
    # Participant Holdout: 20% unseen participants, same 40 items
    participant_holdout_df = df[(df['participant_id'].isin(holdout_p)) & (df['item_id'].isin(train_items))]
    
    # Item Holdout: completely unseen items (evaluate on the holdout participants for true zero-shot)
    # Using holdout participants to prevent data leakage of participant latent structures
    item_holdout_df = df[(df['participant_id'].isin(holdout_p)) & (df['item_id'].isin(holdout_items))]
    
    # Create data directory if missing
    os.makedirs("data", exist_ok=True)
    
    print(f"\nSaving Train (rows: {len(train_df)})...")
    train_df.to_parquet("data/train.parquet", index=False)
    
    print(f"Saving Participant Holdout (rows: {len(participant_holdout_df)})...")
    participant_holdout_df.to_parquet("data/participant_holdout.parquet", index=False)
    
    print(f"Saving Item Holdout (rows: {len(item_holdout_df)})...")
    item_holdout_df.to_parquet("data/item_holdout.parquet", index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()
