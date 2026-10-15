#!/bin/bash
set -e

echo "======================================"
echo "Training LatentLlama Model (1 epoch)..."
echo "======================================"
conda run -n llm_train python train_llama.py --epochs 1

echo "======================================"
echo "Evaluating LatentLlama Model..."
echo "======================================"
conda run -n llm_train python evaluate_llama.py > llama_eval.txt
cat llama_eval.txt

echo "Llama Experiments Complete!"
