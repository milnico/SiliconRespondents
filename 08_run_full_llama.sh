#!/bin/bash
echo "Starting FULL Llama training on Split A (600,000+ items)..."
echo "This will take approximately 10 hours. Logs are written to training_out.log"

conda run -n llm_train python train_llama.py \
    --train_data "split_A_train.parquet" \
    --batch_size 16 \
    --lr 2e-5 \
    > training_out.log 2>&1

echo "Training complete! Model saved to checkpoints/llama_film_model.pt"
