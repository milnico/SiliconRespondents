import pandas as pd

def main():
    print("Loading data.csv...")
    data = pd.read_csv("data.csv")
    print(f"Loaded {len(data)} responses.")
    
    print("Loading latent_scores.csv...")
    latents = pd.read_csv("latent_scores.csv")
    
    print("Parsing codebook.txt...")
    codebook_items = []
    with open("codebook.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) > 1:
                item_id = parts[0].strip()
                # Use rsplit to handle commas inside the item text
                rest = parts[1].rsplit(",", 1)
                item_text = rest[0].strip().strip('"')
                reverse_str = rest[1].strip() if len(rest) > 1 else "false"
                reverse = reverse_str.lower() == "true"
                factor = item_id[0] 
                codebook_items.append({
                    "item_id": item_id,
                    "item_text": item_text,
                    "factor": factor,
                    "reverse": reverse
                })
                
    codebook_df = pd.DataFrame(codebook_items)
    
    data["participant_id"] = [f"P{i+1:05d}" for i in range(len(data))]
    
    print("Melting data into long format...")
    item_cols = [c for c in data.columns if c != "participant_id"]
    
    long_data = pd.melt(
        data, 
        id_vars=["participant_id"], 
        value_vars=item_cols, 
        var_name="item_id", 
        value_name="response"
    )
    
    print("Merging with latent scores...")
    long_data = long_data.merge(latents, on="participant_id", how="left")
    
    print("Merging with codebook metadata...")
    long_data = long_data.merge(codebook_df, on="item_id", how="left")
    
    cols = ["participant_id", "O", "C", "E", "A", "N", "item_id", "item_text", "factor", "reverse", "response"]
    long_data = long_data[cols]
    
    long_data = long_data.dropna()
    long_data["response"] = long_data["response"].astype(int)
    
    print("Saving responses_long.parquet...")
    long_data.to_parquet("responses_long.parquet", index=False)
    print("Done! Total rows:", len(long_data))

if __name__ == "__main__":
    main()
