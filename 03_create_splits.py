import pandas as pd
import numpy as np

def main():
    print("Loading responses_long.parquet...")
    df = pd.read_parquet("responses_long.parquet")
    
    participants = df["participant_id"].unique()
    items = df["item_id"].unique()
    
    np.random.seed(42)
    np.random.shuffle(participants)
    np.random.shuffle(items)
    
    num_p_holdout = int(len(participants) * 0.2)
    split_b_participants = set(participants[:num_p_holdout])
    
    # 2 items per dimension for Split C
    df_items = df[["item_id", "factor"]].drop_duplicates()
    split_c_items_list = []
    for factor in df_items["factor"].unique():
        factor_items = df_items[df_items["factor"] == factor]["item_id"].tolist()
        np.random.shuffle(factor_items)
        split_c_items_list.extend(factor_items[:2])
    split_c_items = set(split_c_items_list)
    
    print(f"Total Participants: {len(participants)}, Split B Holdout: {len(split_b_participants)}")
    print(f"Total Items: {len(items)}, Split C Holdout: {len(split_c_items)}")
    
    mask_b_participant = df["participant_id"].isin(split_b_participants)
    mask_c_item = df["item_id"].isin(split_c_items)
    
    train_mask = (~mask_b_participant) & (~mask_c_item)
    split_a = df[train_mask]
    
    split_b = df[mask_b_participant & (~mask_c_item)]
    split_c = df[mask_c_item & (~mask_b_participant)]
    split_bc = df[mask_b_participant & mask_c_item]
    
    print(f"Split A (Train) size: {len(split_a)}")
    print(f"Split B (New Participants) size: {len(split_b)}")
    print(f"Split C (New Items) size: {len(split_c)}")
    print(f"Split D (New Participants & Items) size: {len(split_bc)}")
    
    print("Saving splits...")
    split_a.to_parquet("split_A_train.parquet", index=False)
    split_b.to_parquet("split_B_new_participants.parquet", index=False)
    split_c.to_parquet("split_C_new_items.parquet", index=False)
    split_bc.to_parquet("split_D_new_both.parquet", index=False)
    print("Done!")

if __name__ == "__main__":
    main()
