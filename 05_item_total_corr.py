import pandas as pd
import numpy as np

def compute_item_total_corr(split_name="Split_C", parquet_file="split_C_new_items.parquet"):
    print(f"=== Item-Total Correlation Analysis ({split_name}) ===")
    
    # Load predictions
    film_df = pd.read_csv(f"predictions_film_{split_name}.csv")
    mlp_df = pd.read_csv(f"predictions_mlp_{split_name}.csv")
    
    # Load original dataset to get the latent traits
    data = pd.read_parquet(parquet_file)
    
    # We only need one row per participant for the traits
    traits_df = data[['participant_id', 'O', 'C', 'E', 'A', 'N']].drop_duplicates()
    
    # Merge traits into predictions
    film_df = film_df.merge(traits_df, on='participant_id', how='left')
    mlp_df = mlp_df.merge(traits_df, on='participant_id', how='left')
    
    items = film_df['item_id'].unique()
    
    results = []
    
    for item in items:
        # The first letter of the item_id tells us which factor it belongs to
        factor = item[0] 
        
        film_item_data = film_df[film_df['item_id'] == item]
        mlp_item_data = mlp_df[mlp_df['item_id'] == item]
        
        # Calculate correlations
        # Human correlation (true response vs factor score)
        human_corr = film_item_data['true_resp'].corr(film_item_data[factor])
        
        # FiLM correlation (predicted response vs factor score)
        film_corr = film_item_data['pred_resp'].corr(film_item_data[factor])
        
        # MLP correlation (predicted response vs factor score)
        mlp_corr = mlp_item_data['pred_resp'].corr(mlp_item_data[factor])
        
        results.append({
            'Item': item,
            'Factor': factor,
            'Human_Corr': human_corr,
            'MLP_Corr': mlp_corr,
            'FiLM_Corr': film_corr
        })
        
    res_df = pd.DataFrame(results)
    
    # Print Item-Level Correlations
    print(res_df.to_string(index=False, float_format="%.3f"))
    
    # Print Mean Correlations
    print("\n--- Mean Item-Total Correlations ---")
    print(f"Human Mean Corr: {res_df['Human_Corr'].abs().mean():.3f}")
    print(f"FiLM Mean Corr:  {res_df['FiLM_Corr'].abs().mean():.3f}")
    print(f"MLP Mean Corr:   {res_df['MLP_Corr'].abs().mean():.3f}")

if __name__ == "__main__":
    compute_item_total_corr()
