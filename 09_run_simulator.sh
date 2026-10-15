#!/bin/bash
# Train the full latent-gated, soft-prefix generative simulator.
set -e
echo "Training generative soft-prefix + latent-gated-LoRA simulator (mode=full)..."
echo "Logs -> simulator_training.log"

conda run -n llm_train python train_simulator.py \
    --train_data "data/train.parquet" \
    --val_data "data/item_holdout.parquet" \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --mode "full" \
    --batch_size 16 \
    --epochs 3 \
    --lr 2e-4 \
    > simulator_training.log 2>&1

echo "Training complete! Best checkpoint: checkpoints/simulator_full.pt"

echo "Generating silicon sample on held-out items..."
conda run -n llm_train python silicon_sample.py \
    --checkpoint checkpoints/simulator_full.pt \
    --items_file data/item_holdout.parquet \
    --n 500 --output_file silicon_sample_itemholdout.csv

echo "Evaluating psychometric structure vs humans..."
conda run -n llm_train python 05_simulator_psychometrics.py \
    --human data/item_holdout.parquet \
    --silicon silicon_sample_itemholdout.csv
